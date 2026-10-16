"""Alignment must be lossless for every tokenizer family we support."""

import pytest

from byteify.alignment import align_from_offsets, align_text, discover_special_bytes
from byteify.boundary_merge import merge_boundaries
from byteify.vocab import build_suffix_trie, resolve_token_bytes

BYTE_LEVEL_MODELS = ["jhu-clsp/ettin-encoder-17m", "answerdotai/ModernBERT-base"]
OTHER_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased",
    "xlm-roberta-base",
]

SAMPLES = [
    "Hello World!",
    "Hello_World!",
    "   leading and trailing   ",
    "café naïve",
    "日本語のテキスト",
    "emoji 🙂🎉 here",
    "def f(x): return x**2",
    "DNA: ACGTACGTNNNN",
    "line1\nline2\ttab",
    "mixed 日本語 and English ünïcödé 🙂 12345",
    "  \t\n  ",
    "https://example.com/path?a=1&b=2#frag",
    "ünïcödé ÜNÏCÖDÉ MiXeD CaSe",
    "a",
]


def _load(model_id):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


@pytest.mark.parametrize("model_id", BYTE_LEVEL_MODELS + OTHER_MODELS)
def test_alignment_partitions_the_byte_stream(model_id):
    """Every byte belongs to exactly one patch, and there is one patch per token.

    This is the invariant the stitching loss depends on: if it fails, patch *i* no
    longer corresponds to token *i* and L_E silently measures the wrong thing.
    """
    tok = _load(model_id)
    for text in SAMPLES:
        a = align_text(text, tok)
        assert sum(a.lengths) == a.n_bytes, f"{model_id} / {text!r}"
        assert a.n_patches == len([x for x in a.lengths if x > 0])
        assert all(l >= 0 for l in a.lengths)
        real = bytes(b for b in a.byte_ids if b < 256)
        assert real.decode("utf-8", errors="replace") or not real


@pytest.mark.parametrize("model_id", BYTE_LEVEL_MODELS)
def test_bytelevel_alignment_matches_token_concatenation(model_id):
    """The offset path must agree with concatenating per-token bytes.

    Byte-level BPE tokens spell out the source text verbatim, so the two independent
    routes to a byte stream have to produce identical results. Disagreement would mean
    offsets lie for this tokenizer.
    """
    tok = _load(model_id)
    resolution = resolve_token_bytes(tok)
    specials = discover_special_bytes(tok)
    for text in SAMPLES:
        a = align_text(text, tok)
        token_ids = tok(text, add_special_tokens=True)["input_ids"]
        rebuilt: list[int] = []
        for i in token_ids:
            if i in specials:
                rebuilt.append(specials[i])  # pseudo-byte, >= 256
            else:
                rebuilt.extend(resolution.token_bytes[i])
        assert rebuilt == list(a.byte_ids), f"{model_id} / {text!r}"


@pytest.mark.parametrize("model_id", OTHER_MODELS)
def test_partial_coverage_is_absorbed_not_dropped(model_id):
    """WordPiece and SentencePiece skip whitespace; those bytes must not vanish."""
    tok = _load(model_id)
    text = "Hello World!"
    a = align_text(text, tok, add_special_tokens=False)
    assert a.covered > 0, "expected this tokenizer to leave gaps to absorb"
    assert sum(a.lengths) == len(text.encode("utf-8")), "every byte must be assigned"


@pytest.mark.parametrize("model_id", BYTE_LEVEL_MODELS + OTHER_MODELS)
def test_no_gaps_mean_full_coverage(model_id):
    tok = _load(model_id)
    from byteify.vocab import resolve_token_bytes

    resolution = resolve_token_bytes(tok)
    if resolution.strategy != "bytelevel":
        pytest.skip("only byte-level BPE covers the text completely")
    for text in SAMPLES:
        a = align_text(text, tok)
        assert a.covered == 0


@pytest.mark.parametrize("model_id", BYTE_LEVEL_MODELS + OTHER_MODELS)
def test_special_tokens_become_pseudo_bytes(model_id):
    tok = _load(model_id)
    specials = discover_special_bytes(tok)
    assert len(specials) >= 2
    a = align_text("hi", tok, add_special_tokens=True)
    pseudo = [b for b in a.byte_ids if b >= 256]
    assert pseudo, "special tokens should appear as pseudo-bytes"
    assert a.n_patches == len([l for l in a.lengths if l > 0])


@pytest.mark.parametrize("model_id", BYTE_LEVEL_MODELS + OTHER_MODELS)
def test_suffix_trie_covers_the_vocabulary(model_id):
    tok = _load(model_id)
    trie, resolution = build_suffix_trie(tok)
    assert len(trie) > 1000, "trie looks empty"
    assert resolution.coverage > 0.9, resolution.describe()


def test_vocab_resolution_strategies():
    for model_id, expected in [
        ("jhu-clsp/ettin-encoder-17m", "bytelevel"),
        ("bert-base-uncased", "wordpiece"),
        ("xlm-roberta-base", "sentencepiece"),
    ]:
        tok = _load(model_id)
        resolution = resolve_token_bytes(tok)
        assert resolution.strategy == expected, f"{model_id} -> {resolution.describe()}"


def test_align_from_offsets_handles_trailing_gap():
    """A final token that does not reach the end of the text must still absorb it."""
    text = "abc def"
    a = align_from_offsets(
        text,
        [10, 11],
        [(0, 3), (4, 6)],  # token 1 stops at 'def', leaving 'def' partially trailing
    )
    assert sum(a.lengths) == len(text.encode())
    assert a.n_patches == 2


def test_align_from_offsets_rejects_out_of_range_offsets():
    with pytest.raises(ValueError):
        align_from_offsets("abc", [1], [(0, 99)])


def test_bytelevel_offsets_are_unreliable_for_multibyte():
    """Documents why byte-level BPE must not be aligned via offsets.

    The fast tokenizer reports overlapping duplicate spans for consecutive tokens in
    CJK text, so an offset-driven byte stream duplicates characters. Concatenating
    per-token bytes is the only correct route for this family.
    """
    tok = _load(BYTE_LEVEL_MODELS[0])
    text = "日本語のテキスト"
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [o for o in encoded["offset_mapping"] if o[1] > o[0]]
    assert len(spans) != len(set(spans)), "expected duplicate offsets for this input"
    # The offset route therefore disagrees with the authoritative token route.
    by_offsets = align_from_offsets(text, encoded["input_ids"], encoded["offset_mapping"])
    by_tokens = align_text(text, tok, add_special_tokens=False)
    assert list(by_offsets.byte_ids) != list(by_tokens.byte_ids)
    assert by_tokens.n_bytes == len(text.encode("utf-8"))


def test_merge_boundaries_never_drops_the_final_patch():
    boundaries = (0, 1, 0, 1, 0, 1)
    merged = merge_boundaries(boundaries, keep=[False, False, False])
    assert list(merged) == [0, 0, 0, 0, 0, 1]


def test_merge_boundaries_to_one_patch():
    boundaries = (0, 1, 0, 1, 0, 1)
    merged = merge_boundaries(boundaries, keep=[False, False, True])
    assert sum(merged) == 1


def test_merge_boundaries_validates_length():
    with pytest.raises(ValueError):
        merge_boundaries((0, 1, 0, 1), keep=[True])