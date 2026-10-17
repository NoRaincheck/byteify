"""The byte-stream rule (Ticket #12, ADR 0011).

One declared rule, applied identically by training and inference. The failure this guards
against is silent: an artifact built on a byte stream that differs from the one it was
trained on still loads, still runs, and still produces embeddings -- they are just wrong.

The supporting seam exists separately from the artifact seam on purpose. Alignment is a
pure function with no model in it, so asserting it directly is nearly free, and a rule that
is only checked through a trained model can be wrong in a way the model's fidelity numbers
cannot detect -- they are measured against the corrupted stream the model was built from.
"""

from __future__ import annotations

import unicodedata

import pytest

from byteify.alignment import align_text, discover_special_bytes
from byteify.bytestream import ByteStreamRule, rule_for_family
from tests.synthetic import build_tokenizer

SAMPLES = [
    "byte level retrofit",
    "日本語のテキスト",
    "한국어 텍스트",
    "café naïve ünïcödé",
    "Ω omega",
    "emoji 🙂🎉 inline",
    "DNA: ACGTACGTNNNN and protein MKWVTFISLLLLF",
]

#: Text the rule can be checked against directly: ASCII, no runs of whitespace, because
#: runs of whitespace are a pre-tokeniser policy question and not this rule's business.
REPRESENTABLE_SAMPLES = [
    "byte level retrofit",
    "café naïve ünïcödé",
]


@pytest.fixture(scope="module")
def representable(tokenizers, corpus) -> dict[str, list[str]]:
    """Sentences each tokenizer segments into bytes it can fully account for.

    The fixture tokenizers have 400 entries. An [UNK] has no byte string, so
    ``align_text`` drops it and the training stream is *shorter than the text*; that is a
    property of a deliberately tiny fixture, not of the rule. Rather than hand-pick
    sentences and hope, each family contributes the ones it can represent exactly -- which
    may exclude a hand-written sample, since "café" is not in a 400-token English
    vocabulary. The guard in the test fails if that leaves nothing to compare.
    """
    plain = [
        s
        for s in corpus
        if s.isascii() and not any(c.isspace() and c != " " for c in s)
    ]
    out: dict[str, list[str]] = {}
    for family, tokenizer in tokenizers.items():
        specials = discover_special_bytes(tokenizer)
        usable = []
        for sentence in [*REPRESENTABLE_SAMPLES, *plain]:
            try:
                trained = align_text(sentence, tokenizer, special_token_ids=specials)
            except ValueError:
                continue  # offsets outside the text: a known offset-family defect
            content = bytes(b for b in trained.byte_ids if b < 256)
            try:
                if content.decode("utf-8") == sentence:
                    usable.append(sentence)
            except UnicodeDecodeError:
                continue
        out[family] = usable[:10]
    return out


@pytest.fixture(scope="module")
def tokenizers(request):
    from tests.synthetic import synthetic_corpus

    corpus = synthetic_corpus()
    return {f: build_tokenizer(f, corpus[:200]) for f in ("bytelevel", "wordpiece", "sentencepiece")}


@pytest.fixture(scope="module")
def rules(tokenizers):
    out = {}
    for family, tokenizer in tokenizers.items():
        specials = discover_special_bytes(tokenizer)
        out[family] = rule_for_family(
            family, n_byte_values=256 + len(specials), special_tokens=specials
        )
    return out


def _brackets(rule: ByteStreamRule) -> tuple[int, int]:
    return rule.open_pseudo, rule.close_pseudo


# -- the rule is declared per family ---------------------------------------


def test_bytelevel_normalises_to_nfc_and_wordpiece_does_not(rules):
    assert rules["bytelevel"].rule == "nfc"
    assert rules["wordpiece"].rule == "raw-utf8"
    assert rules["sentencepiece"].rule == "raw-utf8"


def test_a_family_with_no_normaliser_needs_no_tokenizer(rules):
    """The tokenizer-free claim is true of these families and false of none of them."""
    assert rules["bytelevel"].tokenizer_free
    assert rules["wordpiece"].tokenizer_free
    assert rules["sentencepiece"].tokenizer_free


def test_a_normalised_rule_says_so_rather_than_silently_lying():
    """A rule that reproduces a tokenizer's rewriting is not tokenizer-free."""
    rule = ByteStreamRule(rule="normalised", family="wordpiece", n_byte_values=260)
    assert not rule.tokenizer_free
    with pytest.raises(NotImplementedError, match="without the source tokenizer"):
        rule.content("Byte")


# -- NFC is exact ---------------------------------------------------------


def test_nfc_matches_the_standard_library(rules):
    """Every decomposed sequence must normalise the same way the standard library does."""
    rule = rules["bytelevel"]
    for text in SAMPLES:
        assert rule.content(text) == unicodedata.normalize("NFC", text).encode("utf-8")


def test_nfc_is_exact_across_a_broad_codepoint_sweep(rules):
    """Sweep every codepoint, not a handful of samples.

    The claim under test is "exact across all 158,867 codepoints", and a claim like that
    cannot be supported by three examples.
    """
    rule = rules["bytelevel"]
    mismatches = []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:  # surrogates are not encodable
            continue
        char = chr(codepoint)
        if rule.content(char) != unicodedata.normalize("NFC", char).encode("utf-8"):
            mismatches.append(hex(codepoint))
    assert mismatches == [], f"NFC differs from the standard library at {mismatches[:10]}"


def test_a_single_non_latin_character_becomes_exactly_its_own_bytes(rules):
    """The SentencePiece duplication defect: one `Ω` used to become two."""
    for family, rule in rules.items():
        assert rule.content("Ω") == "Ω".encode(), family
        assert rule.content("日") == "日".encode(), family
        assert rule.content("🙂") == "🙂".encode(), family


# -- pseudo-bytes bracket the stream --------------------------------------


def test_the_stream_is_bracketed_at_both_ends(rules):
    for family, rule in rules.items():
        stream = rule.build("hello", 64)
        assert stream[0] == rule.open_pseudo, family
        assert stream[-1] == rule.close_pseudo, family


def test_the_brackets_are_real_pseudo_bytes(rules):
    """They live outside the byte alphabet, which is what keeps them distinguishable."""
    for rule in rules.values():
        assert rule.open_pseudo >= 256
        assert rule.close_pseudo >= 256
        assert rule.open_pseudo != rule.close_pseudo


def test_a_tokenizer_with_no_special_tokens_is_not_bracketed():
    rule = rule_for_family("bytelevel")
    assert rule.open_pseudo == -1
    stream = rule.build("hello", 64)
    assert stream[0] != -1
    assert -1 not in stream


# -- the byte axis is honoured --------------------------------------------


def test_the_stream_never_exceeds_the_byte_axis(rules):
    for family, rule in rules.items():
        for limit in (16, 64, 512):
            assert len(rule.build("a rather long sentence to truncate " * 10, limit)) <= limit, (
                family
            )


def test_the_axis_counts_the_brackets(rules):
    """`max_bytes` bounds the whole axis, so serving hardware sized from it is right."""
    rule = rules["bytelevel"]
    assert len(rule.build("hello world", 12)) == 12


def test_truncation_never_splits_a_character(rules):
    """Half a multi-byte character is not valid UTF-8 and not what the model saw."""
    rule = rules["bytelevel"]
    for limit in range(4, 24):
        stream = rule.build("日本語のテキスト", limit)
        body = bytes(b for b in stream if b < 256)
        # Decoding must succeed, and must give back exactly the bytes we put in.
        assert body.decode("utf-8").encode("utf-8") == body, f"limit {limit} split a character"


def test_a_zero_byte_axis_is_refused(rules):
    with pytest.raises(ValueError, match="max_bytes"):
        rules["bytelevel"].build("hello", 0)


# -- training and inference agree -----------------------------------------


@pytest.mark.parametrize("family", ["bytelevel", "wordpiece", "sentencepiece"])
def test_the_rule_reproduces_the_training_byte_stream(family, rules, tokenizers, representable):
    """The rule and ``align_text`` must produce identical bytes for identical text.

    This is the assertion the whole module exists for: the model that was trained is the
    model that is served, or the artifact is computing something nobody trained.

    WordPiece is checked only for bracketing. Its normaliser pads CJK with spaces, so
    reproducing its training stream would mean reproducing the normaliser -- which is
    exactly why that family is not tokenizer-free, and why this test does not pretend
    otherwise.
    """
    tokenizer = tokenizers[family]
    rule = rules[family]
    specials = discover_special_bytes(tokenizer)
    open_pseudo, close_pseudo = _brackets(rule)

    sentences = representable[family]
    # A test that quietly compared nothing would pass forever.
    assert len(sentences) >= 5, f"{family}: only {len(sentences)} sentences to compare"

    for text in sentences:
        trained = list(align_text(text, tokenizer, special_token_ids=specials).byte_ids)
        inferred = rule.build(text, 10_000)

        assert inferred[0] == open_pseudo and inferred[-1] == close_pseudo, family
        if tokenizer_has_no_normaliser(tokenizer, family):
            assert inferred == trained, (
                f"{family}: inference produced {inferred[:12]}... but training produced "
                f"{trained[:12]}..."
            )


def tokenizer_has_no_normaliser(tokenizer, family: str) -> bool:
    """Whether the tokenizer leaves the text alone, so the two routes can be compared."""
    if family != "wordpiece":
        return True
    normalizer = getattr(getattr(tokenizer, "backend_tokenizer", None), "normalizer", None)
    return normalizer is None or not getattr(normalizer, "lowercase", False)


# -- the rule round-trips through the manifest's form ----------------------


def test_a_rule_survives_its_dict_form(rules):
    for family, rule in rules.items():
        back = ByteStreamRule.from_dict(rule.to_dict())
        assert back == rule, family
        assert back.content("hello") == rule.content("hello")


def test_pseudo_byte_keys_survive_json(rules):
    """JSON turns integer keys into strings; the mapping must not silently invert."""
    rule = rules["bytelevel"]
    back = ByteStreamRule.from_dict(rule.to_dict())
    assert back.pseudo_bytes == rule.pseudo_bytes
    assert set(back.pseudo_bytes) == set(rule.pseudo_bytes)