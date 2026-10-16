"""Reconstructing a byteified encoder, in one place.

The dimensions a byteified encoder needs -- hidden width, byte-value count, vocabulary
size -- are all derived from the source model and its tokenizer, but the derivation was
copied into six entry points with slightly different arguments. Three of those call sites
went further and hardcoded the local encoder's own architecture, so a checkpoint trained
with a non-default local encoder could not be loaded back: the reconstruction built a
different shape and the load failed on a size mismatch.

The local encoder's architecture is therefore data, recorded in the checkpoint and read
back, rather than a constant at the call site.
"""

from __future__ import annotations

import torch

from byteify.source import LocalEncoderSpec, build_byteified_encoder, load_byteified_encoder


def _save(path, encoder, spec, model_id):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "config": {**spec.to_config(), "model": model_id},
            "step": 0,
        },
        path,
    )


def test_a_non_default_local_encoder_round_trips_from_its_checkpoint(
    tmp_path, tiny_source_encoder, tiny_source_tokenizer
):
    """The defect: a checkpoint whose local encoder was not the default would not load."""
    spec = LocalEncoderSpec(layers=2, window=8, ffn_hidden=32, backend="attention")
    encoder = build_byteified_encoder(tiny_source_encoder, tiny_source_tokenizer, spec)
    path = tmp_path / "ckpt.pt"
    _save(path, encoder, spec, "synthetic/tiny")

    reloaded = load_byteified_encoder(path, tiny_source_encoder, tiny_source_tokenizer)

    # The state-dict comparison below is what actually proves the architecture matched:
    # a different window or feed-forward width changes tensor shapes, and the load fails.
    assert len(reloaded.local_encoder.ffns) == 2
    for (name, expected), (_, got) in zip(
        encoder.state_dict().items(), reloaded.state_dict().items()
    ):
        assert torch.equal(expected, got), f"{name} did not survive the round trip"


def test_a_reloaded_encoder_produces_the_same_embeddings(
    tmp_path, tiny_source_encoder, tiny_source_tokenizer
):
    from byteify.inference import byte_ids_for_inference

    spec = LocalEncoderSpec(layers=1, window=16, ffn_hidden=64)
    encoder = build_byteified_encoder(tiny_source_encoder, tiny_source_tokenizer, spec).eval()
    path = tmp_path / "ckpt.pt"
    _save(path, encoder, spec, "synthetic/tiny")

    reloaded = load_byteified_encoder(path, tiny_source_encoder, tiny_source_tokenizer).eval()

    ids, expanded, mask = byte_ids_for_inference(
        tiny_source_tokenizer, "byte level retrofit", 64
    )
    with torch.no_grad():
        assert torch.allclose(encoder(ids, expanded, mask).byte_reps, reloaded(ids, expanded, mask).byte_reps)


def test_dimensions_are_derived_not_declared(tiny_source_encoder, tiny_source_tokenizer):
    """A caller states the model; the byte axis and vocabulary come from the tokenizer."""
    encoder = build_byteified_encoder(tiny_source_encoder, tiny_source_tokenizer)
    assert encoder.embed.byte.weight.shape[1] == tiny_source_encoder.d_model
    assert encoder.embed.suffix.weight.shape[0] > 0
    # Pseudo-bytes reserve room above the 256 real byte values.
    assert encoder.embed.byte.weight.shape[0] > 256


def test_the_spec_round_trips_through_the_checkpoint_format():
    original = LocalEncoderSpec(layers=3, window=64, ffn_hidden=128, backend="gru")
    assert LocalEncoderSpec.from_config(original.to_config()) == original