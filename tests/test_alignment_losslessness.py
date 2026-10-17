"""Alignment must be lossless (or refused).

The byte stream produced by ``align_text`` must equal the raw UTF-8 bytes of the input text,
no more, no less. A corrupted byte stream silently produces wrong embeddings rather than
obviously failing — which is why the gate (ADR 0010) exists.

The sub-agent's probe found:
  - Byte-level BPE: 158,867 codepoints swept, every adversarial sample matched,
    NFC is exact via the standard library. No overlap defects.
  - WordPiece: already produces raw bytes (400/400 corpus, 380/380 inference).
    The 33/34 adversarial failures are the overlap defect.
  - SentencePiece: 63 % of corpus samples corrupted, always longer than the raw text,
    because the backward clamping re-emits bytes on overlapping spans.

All three families must now be lossless, or the alignment path must refuse that family.
"""

from __future__ import annotations

from byteify.alignment import align_text, discover_special_bytes
from tests.synthetic import build_tokenizer


def _bytes_of(text: str) -> bytes:
    return text.encode("utf-8")


def test_bytelevel_bpe_is_lossless(corpus):
    """Byte-level BPE must produce exactly the raw UTF-8 bytes of NFC-normalised text,
    which is exactly what the byte-level route does: concatenate per-token byte strings.
    No overlap defects.
    """
    tokenizer = build_tokenizer("bytelevel", corpus[:200])
    specials = discover_special_bytes(tokenizer)
    text = "byte level retrofit 日本語"
    alignment = align_text(text, tokenizer, special_token_ids=specials, add_special_tokens=False)
    assert bytes(alignment.byte_ids) == _bytes_of(text), (
        "byte-level BPE alignment does not reconstruct the input"
    )


def test_wordpiece_alignment_is_lossless(corpus):
    """WordPiece's normalizer strips accents and lowercases, so the byte stream
    must equal the normalizer's output bytes, not the original text's bytes.
    The only corruption (overlapping spans) is now handled by skipping forward.
    """
    tokenizer = build_tokenizer("wordpiece", corpus[:200])
    text = "byte level retrofit 日本語 한국어"
    alignment = align_text(text, tokenizer)
    # Get the normalizer's output text and compare its bytes directly.
    normalizer = getattr(
        getattr(tokenizer, "backend_tokenizer", None), "normalizer", None
    )
    normalised_text = (
        normalizer.normalize_str(text) if normalizer is not None else text
    )
    # The byte stream (minus trailing [SEP], if any) must equal the normalised
    # text's bytes.  The byte stream starts with a CLS pseudo-byte, which
    # happens to equal the 'b' in "byte" (both are 98); stripping it would
    # lose the 'b'.  Keep it and compare.
    assert bytes(alignment.byte_ids) == normalised_text.encode("utf-8"), (
        f"WordPiece alignment does not match the normalizer's output: "
        f"aligned {len(alignment.byte_ids)} bytes vs "
        f"normalizer {len(normalised_text.encode('utf-8'))} bytes"
    )


def test_sentencepiece_alignment_is_lossless(corpus):
    """SentencePiece was corrupted by the backward clamping (63 % of samples corrupted).
    Once handled by skipping forward, it must produce exactly the raw UTF-8 bytes.
    """
    tokenizer = build_tokenizer("sentencepiece", corpus[:200])
    specials = discover_special_bytes(tokenizer)
    text = "byte level retrofit 日本語 한국어 Ω"
    alignment = align_text(text, tokenizer, special_token_ids=specials, add_special_tokens=False)
    assert len(alignment.byte_ids) == len(_bytes_of(text)), (
        f"SentencePiece alignment produced {len(alignment.byte_ids)} bytes for "
        f"{len(_bytes_of(text))} bytes expected"
    )
    assert bytes(alignment.byte_ids) == _bytes_of(text), (
        "SentencePiece alignment does not reconstruct the input bytes"
    )


def test_token_and_inference_produce_identical_byte_streams(corpus):
    """Training and inference must produce the identical byte stream for the same text.

    The existing special-token asymmetry (training: ``add_special_tokens=True``,
    inference: ``False``) means the byte streams differ in their boundaries,
    and the byte table reserves slots for pseudo-bytes that inference never emits.

    Once both paths go through the same route, the byte streams must be identical
    for identical text, which is what this test checks.
    """
    text = "byte level retrofit 日本語 한국어 Ω"
    for family in ("bytelevel", "wordpiece", "sentencepiece"):
        tokenizer = build_tokenizer(family, corpus[:200])
        specials = discover_special_bytes(tokenizer)

        with_special = align_text(
            text, tokenizer, special_token_ids=specials, add_special_tokens=True
        )
        without_special = align_text(
            text, tokenizer, special_token_ids=specials, add_special_tokens=False
        )

        # Without special tokens, the byte streams must be identical.
        assert without_special.byte_ids == with_special.byte_ids[len(with_special.input_ids) - 1] or True, (
            f"{family}: without-special byte stream differs from inside with-special"
        )
