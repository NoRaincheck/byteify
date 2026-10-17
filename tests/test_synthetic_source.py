"""The synthetic source model must be a real model, not a stand-in.

These are the fixtures every other test leans on, so they are worth checking directly: a
model that routes through the source-model interface, and a tokenizer that genuinely
segments text into bytes. If either silently degrades to something weaker, every later
test is testing the wrong thing.
"""

from __future__ import annotations

import pytest


def _characters(text: str) -> str:
    return "".join(text.split())


def test_synthetic_model_routes_through_the_source_interface(tiny_source_encoder):
    source = tiny_source_encoder
    assert source.family == "modernbert"
    assert source.d_model == 32
    assert source.n_layers == 2
    assert source.embedding_matrix().shape == (source.vocab_size, 32)


def test_synthetic_model_runs_layers_over_supplied_representations(tiny_source_encoder):
    import torch

    source = tiny_source_encoder
    reps = torch.randn(2, 5, source.d_model)
    mask = torch.ones(2, 5, dtype=torch.long)

    assert source.forward_layers(reps, mask).shape == (2, 5, source.d_model)
    assert source.forward_layers(reps, mask, n_layers=1).shape == (2, 5, source.d_model)


def test_every_synthetic_tokenizer_segments_text_it_can_reassemble(tiny_tokenizers):
    """Each tokenizer must really segment, really round-trip, and report sane offsets.

    Not the alignment-losslessness check — that belongs at the byte-stream seam, against
    the library's own alignment path. This is the weaker, earlier guarantee: the fixture
    is a working tokenizer and not a stub.
    """
    text = "byte level retrofit 日本語"
    for family, tokenizer in tiny_tokenizers.items():
        ids = tokenizer(text)["input_ids"]
        assert len(ids) > 1, f"{family} collapsed the whole input into one token"

        offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        # Special tokens report (0, 0) wherever they sit, so the ordering guarantee is
        # about the tokens that actually cover text. Including them would fail on a
        # trailing [SEP].
        starts = [start for start, end in offsets if end > start]
        assert starts == sorted(starts), f"{family} produced out-of-order offsets"
        assert all(0 <= start <= end <= len(text) for start, end in offsets), (
            f"{family} produced an offset outside the input"
        )

        # WordPiece's normaliser pads CJK with spaces, so round-tripping is exact for
        # characters but not for whitespace. Compare characters; the byte-level seam is
        # where whitespace policy is actually asserted.
        assert _characters(tokenizer.decode(ids, skip_special_tokens=True)) == _characters(text), (
            f"{family} could not reassemble the text it segmented"
        )


def test_synthetic_tokenizers_resolve_a_byte_string_per_token(tiny_tokenizers):
    from byteify.vocab import resolve_token_bytes

    for family, tokenizer in tiny_tokenizers.items():
        resolution = resolve_token_bytes(tokenizer, use_cache=False)
        assert resolution.strategy != "literal", f"{family} fell back to literal matching"
        assert resolution.coverage > 0.9, f"{family} only resolved {resolution.coverage:.1%}"


@pytest.mark.parametrize("family", ["modernbert", "bert"])
def test_both_architectures_are_available(tiny_source_dir, family):
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    model = AutoModel.from_pretrained(tiny_source_dir[family])
    assert HuggingFaceSourceEncoder(model).n_layers == 2


def test_the_synthetic_model_is_small_enough_to_byteify_in_a_test(
    tiny_source_encoder, tiny_source_tokenizer
):
    """The point of the fixture: a whole retrofit, in a unit test, in no time at all."""
    from byteify.alignment import discover_special_bytes
    from byteify.model import ByteifiedEncoder, LocalContextEncoder
    from byteify.vocab import resolve_token_bytes

    source = tiny_source_encoder
    resolution = resolve_token_bytes(tiny_source_tokenizer, use_cache=False)
    specials = discover_special_bytes(tiny_source_tokenizer)
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    encoder.embed.load_suffix_from(source.embedding_matrix())

    parameters = sum(p.numel() for p in encoder.parameters())
    assert parameters < 200_000, "the synthetic model is no longer cheap to byteify"


def test_a_byteified_synthetic_model_produces_an_embedding(
    tiny_source_encoder, tiny_source_tokenizer
):
    """End to end on the fixture: text in, embedding out, no network, no real checkpoint."""
    import torch

    from byteify.alignment import discover_special_bytes
    from byteify.inference import ByteifiedEmbedder
    from byteify.model import ByteifiedEncoder, LocalContextEncoder
    from byteify.vocab import resolve_token_bytes

    source = tiny_source_encoder
    resolution = resolve_token_bytes(tiny_source_tokenizer, use_cache=False)
    specials = discover_special_bytes(tiny_source_tokenizer)
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    encoder.embed.load_suffix_from(source.embedding_matrix())

    embedder = ByteifiedEmbedder(encoder, source, n_patches=8)
    embedding = embedder.encode(tiny_source_tokenizer, ["byte level retrofit 日本語"])

    assert embedding.shape == (1, source.d_model)
    assert torch.isfinite(embedding).all()