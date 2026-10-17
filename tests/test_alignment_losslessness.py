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


def _brackets(specials: dict[int, int]) -> tuple[int, int]:
    """The two pseudo-bytes that bracket the stream, in discovery order.

    ``discover_special_bytes`` probes class, separator, pad, mask in that order, so the
    first two values it assigns are the ones that frame the byte axis.
    """
    values = list(specials.values())
    return values[0], values[1]


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
    specials = discover_special_bytes(tokenizer)
    open_pseudo, close_pseudo = _brackets(specials)
    text = "byte level retrofit 日本語 한국어"
    alignment = align_text(text, tokenizer, special_token_ids=specials)
    # Get the normalizer's output text and compare its bytes directly.
    normalizer = getattr(
        getattr(tokenizer, "backend_tokenizer", None), "normalizer", None
    )
    normalised_text = (
        normalizer.normalize_str(text) if normalizer is not None else text
    )
    # Pseudo-bytes bracket the stream at both ends, and they are not bytes at all, so the
    # comparison has to take them off first. Comparing the whole sequence used to be
    # impossible here -- it only worked because the fixture emitted no pseudo-bytes.
    bracketed = list(alignment.byte_ids)
    expected = [open_pseudo] + list(normalised_text.encode("utf-8")) + [close_pseudo]
    assert bracketed == expected, (
        f"WordPiece alignment does not match the normalizer's output: "
        f"aligned {len(bracketed)} bytes vs normalizer {len(expected)} expected"
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

    Training runs with ``add_special_tokens=True`` and inference with ``False``, so the two
    differ by exactly the bracketing pseudo-bytes -- the class and separator positions that
    hold the byte axis in sync with the token axis. Inside the brackets they must agree
    byte for byte, or the model that was trained is not the model that is served.

    This test previously ended in ``or True``, so it asserted nothing at all.
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

        bracketed = list(with_special.byte_ids)
        # The first and last positions are pseudo-bytes; the content between them is the
        # stream both paths must agree on.
        assert bracketed[1:-1] == list(without_special.byte_ids), (
            f"{family}: the bracketed byte stream differs from the unbracketed one "
            f"inside the brackets"
        )
        assert len(bracketed) == len(without_special.byte_ids) + 2, (
            f"{family}: expected exactly two bracketing pseudo-bytes, got "
            f"{len(bracketed) - len(without_special.byte_ids)}"
        )
