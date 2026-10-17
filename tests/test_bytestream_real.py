"""The byte-stream rule against the real anchor model.

The fixture tier proves the rule is consistent with a working tokenizer. This tier proves
it against the tokenizer the artifact was actually trained with, because the fixture's
400-token vocabulary cannot represent most real text and so cannot exercise the paths that
matter -- decomposed accents, emoji, surrogate pairs, and characters the real vocabulary
covers with multi-byte tokens.

Marked as the slower tier by virtue of loading a real model. Kept, not replaced.
"""

from __future__ import annotations

import unicodedata

import pytest
from transformers import AutoTokenizer

from byteify.alignment import align_text, discover_special_bytes
from byteify.bytestream import rule_for_family

ANCHOR = "jhu-clsp/ettin-encoder-17m"

SAMPLES = [
    "the quick brown fox jumps over the lazy dog",
    "def f(x): return x + 1",
    "café naïve ünïcödé",
    "Ω omega",
    "日本語のテキスト",
    "한국어 텍스트",
    "emoji 🙂🎉 inline",
    "   multiple   spaces   ",
    "Ω",
    "Ω",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(ANCHOR)


@pytest.fixture(scope="module")
def rule(tokenizer):
    specials = discover_special_bytes(tokenizer)
    return rule_for_family(
        "bytelevel", n_byte_values=256 + len(specials), special_tokens=specials
    )


def test_the_anchor_tokenizer_declares_a_tokenizer_free_rule(rule):
    assert rule.rule == "nfc"
    assert rule.tokenizer_free


def test_the_anchor_bracket_pseudo_bytes_are_the_first_two_discovered(tokenizer, rule):
    specials = discover_special_bytes(tokenizer)
    assert (rule.open_pseudo, rule.close_pseudo) == tuple(list(specials.values())[:2])


@pytest.mark.parametrize("text", SAMPLES)
def test_inference_and_training_agree_byte_for_byte(text, tokenizer, rule):
    """The rule and the training path must produce the identical stream.

    Not "close". Not "the same length". The same bytes, because the byte stream is what
    the model was trained on and any difference is an artifact computing something nobody
    trained.
    """
    specials = discover_special_bytes(tokenizer)
    trained = list(align_text(text, tokenizer, special_token_ids=specials).byte_ids)
    inferred = rule.build(text, 100_000)
    assert inferred == trained


@pytest.mark.parametrize("text", SAMPLES)
def test_the_inferred_stream_is_the_texts_bytes_between_brackets(text, rule):
    body = [b for b in rule.build(text, 100_000) if b < 256]
    assert bytes(body).decode("utf-8") == unicodedata.normalize("NFC", text)


def test_an_unknown_character_is_absorbed_not_emitted_as_a_pseudo_byte(tokenizer, rule):
    """A character the vocabulary cannot represent must not become a value in the stream.

    Regression: `[UNK]` was in the special-token map, so an unrepresentable character
    produced a pseudo-byte *mid-stream* -- a value that is not a byte of the text, at
    exactly the positions the tokenizer struggled. The stream silently disagreed with the
    input, and the artifact looked fine.
    """
    specials = discover_special_bytes(tokenizer)
    unk_id = tokenizer.unk_token_id
    assert unk_id is not None
    assert unk_id in specials, "the anchor must actually have an [UNK] to be a regression"

    # A rare CJK extension character is very unlikely to be in a 50k English vocabulary.
    text = "\U00020000" * 4
    trained = list(align_text(text, tokenizer, special_token_ids=specials).byte_ids)

    # The brackets are pseudo-bytes by design; the body must contain nothing but bytes.
    assert trained[0] == rule.open_pseudo and trained[-1] == rule.close_pseudo
    assert all(b < 256 for b in trained[1:-1]), (
        f"a pseudo-byte leaked into the body of the stream: {trained}"
    )
    assert bytes(trained[1:-1]) == text.encode("utf-8")


def test_the_reserved_pseudo_bytes_are_never_emitted(rule):
    """ADR 0011: the remaining reserved byte values no path can emit."""
    emitted_values = {rule.open_pseudo, rule.close_pseudo}
    for text in SAMPLES:
        stream = rule.build(text, 100_000)
        pseudo = {b for b in stream if b >= 256}
        assert pseudo <= emitted_values, f"{text!r} emitted {pseudo}"


def test_the_anchor_tokenizer_has_no_unk_in_the_byte_table_size(tokenizer):
    """Dropping the [UNK] pseudo-byte from emission must not shrink the byte table.

    The values stay reserved so an existing checkpoint's byte table still matches its
    manifest. If this ever changes, every artifact built before it becomes unloadable.
    """
    specials = discover_special_bytes(tokenizer)
    assert len(specials) > 2
    assert 256 + len(specials) > 256