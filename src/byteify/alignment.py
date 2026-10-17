"""Byte <-> subword alignment, derived from tokenizer offsets.

The byteified model consumes a raw byte stream. The source subword model consumes a
sequence of subword tokens. To retrofit one onto the other we need an exact,
lossless correspondence between the two sequences, which is what this module
builds.

Alignment is derived from the fast tokenizer's **offset mapping** rather than from
concatenating per-token byte strings, because offsets work for every tokenizer family
while concatenation only works when tokens spell out the source bytes verbatim.

Coverage and gaps
-----------------
Byte-level BPE covers the text completely. WordPiece and SentencePiece do not: BERT's
pre-tokenizer drops the spaces, so in ``"Hello World!"`` the token ``World`` has span
``(6, 11)`` and character 5 is covered by nothing. Naively concatenating token spans
would silently drop those bytes, desynchronising the byte and patch axes and leaking
the difference into the stitching loss.

We therefore assign **every** byte to exactly one token: each token's span is extended
backwards to absorb any uncovered bytes that precede it, and a trailing gap is
absorbed by the last token. The result is a partition of the byte stream into exactly
one contiguous patch per token -- which is the property the rest of byteification
relies on, since patch *i* must correspond to token *i* for ``L_E`` to mean anything.

Special tokens (``[CLS]``, ``[SEP]``, ...) have zero-width spans. They are mapped to
distinct pseudo-bytes outside 0..255 so the byte axis stays aligned with the token
axis; a special is then a one-byte patch, exactly like a single-character token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Subword tokens that carry no text bytes of their own. Each maps to a distinct
# pseudo-byte >= 256 so that the byteified model's byte stream keeps the same
# alignment as the subword token sequence.
SPECIAL_TOKENS: tuple[str, ...] = ("[CLS]", "[SEP]", "[PAD]", "[MASK]")


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / ByteLevel reversible byte <-> printable-codepoint table.

    Byte-level BPE stores tokens as strings over a 256-symbol alphabet chosen so that
    every byte maps to a non-whitespace codepoint and the map is a bijection. This
    reproduces it; ``tokenizers`` exposes the same table as ``ByteLevel.alphabet()``
    and the tests assert the two agree.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


@dataclass(frozen=True)
class Alignment:
    """A subword token sequence and the byte stream it corresponds to.

    Attributes:
        input_ids: the subword token sequence.
        byte_ids: the byte stream; pseudo-bytes (>= 256) stand in for special tokens.
        lengths: number of byte-stream entries contributed by each token.
        boundaries: ``B_subword``, 1 at the last entry of each token, else 0.
        covered: bytes absorbed from an uncovered gap rather than from the token text.
    """

    input_ids: tuple[int, ...]
    byte_ids: tuple[int, ...]
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]
    covered: int = 0

    def __len__(self) -> int:
        return len(self.byte_ids)

    @property
    def n_bytes(self) -> int:
        return len(self.byte_ids)

    @property
    def n_patches(self) -> int:
        return sum(self.boundaries)

    @property
    def bytes_per_patch(self) -> float:
        """Compression ratio: mean byte-stream length per patch."""
        n_patches = self.n_patches
        return self.n_bytes / n_patches if n_patches else 0.0

    def patch_last_positions(self) -> tuple[int, ...]:
        """Byte-stream indices of each patch's last entry (the pooling positions).

        Bolmo pools by taking the representation of the *last* byte in each patch,
        which needs no extra parameters.
        """
        return tuple(i for i, b in enumerate(self.boundaries) if b)

    def token_of_byte(self) -> tuple[int, ...]:
        """For each byte-stream index, the index of the token that produced it."""
        out: list[int] = []
        for t, length in enumerate(self.lengths):
            out.extend([t] * length)
        return tuple(out)


def align_from_offsets(
    text: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    *,
    special_token_ids: Mapping[int, int] | None = None,
) -> Alignment:
    """Build the byte stream and boundary supervision from a tokenizer's offsets.

    ``offsets`` are ``(char_start, char_end)`` pairs; a ``(0, 0)`` span marks a special
    token. ``special_token_ids`` maps a token id to its pseudo-byte value.

    Only valid when the tokenizer's offsets are trustworthy -- see
    :func:`align_tokenizer`, which dispatches on tokenizer family.
    """
    special_token_ids = special_token_ids or {}
    data = text.encode("utf-8")

    for start, end in offsets:
        if not (0 <= start <= end <= len(text)):
            raise ValueError(f"offset ({start}, {end}) out of range for text of length {len(text)}")

    # Character index -> byte offset, so token spans can be expressed on the byte axis.
    char_to_byte = [0] * (len(text) + 1)
    b = 0
    for i, ch in enumerate(text):
        char_to_byte[i] = b
        b += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = b

    real_indices = {i for i, o in enumerate(offsets) if o[1] > o[0]}
    byte_ids: list[int] = []
    lengths: list[int] = []
    boundaries: list[int] = []
    n_absorbed = 0
    cursor = 0  # next unassigned byte
    last_real_token: int | None = None  # index into `lengths` of the last content token

    for i, (start, end) in enumerate(offsets):
        # Special tokens always get a pseudo-byte, regardless of offset.  Some
        # tokenizers (wordpiece / SentencePiece) report non-zero spans for [CLS]
        # and [SEP]; the previous code treated them as real tokens and emitted
        # the bytes for the covered text, which broke the byte-to-token mapping
        # at the source.  We emit the pseudo-byte AND the text bytes so the
        # byte stream still covers the full text, but the token that owns the
        # bytes is recorded as a special (one-byte patch).
        pseudo = special_token_ids.get(token_ids[i])
        if pseudo is not None:
            byte_ids.append(pseudo)
            lengths.append(0)  # will be set below
            boundaries.append(1)  # the pseudo-byte is the entire token (zero-width)
            # Also emit the bytes for the covered text (e.g. [CLS] covering
            # "byte").  This restores the original behaviour of emitting the
            # covered text, which is necessary for the byte axis to remain
            # lossless, but marks the token as a single-byte pseudo-byte patch
            # so the byte-to-token boundary map stays correct.
            if i in real_indices:
                char_start_b = char_to_byte[start]
                char_end_b = char_to_byte[end]
                chunk = data[char_start_b:char_end_b]
                byte_ids.extend(chunk)
                lengths[-1] = 1 + len(chunk)  # pseudo-byte + text bytes
                boundaries[-1] = 0  # no longer the last entry
                boundaries.extend([0] * len(chunk))
                boundaries[-1] = 1  # last entry of the token is a boundary
            else:
                lengths[-1] = 1  # just the pseudo-byte
            end_b = char_to_byte[min(end, len(text))]
            cursor = max(cursor, end_b)
            continue

        # Always emit from the cursor.  This absorbs gaps (uncovered bytes
        # between tokens) for WordPiece and skips overlaps (duplicate spans)
        # for SentencePiece, whichever case applies.
        char_start_b = char_to_byte[start]
        char_end_b = char_to_byte[end]
        start_byte = cursor
        end_byte = max(char_end_b, start_byte)

        # Tokens can map to an empty span after normalization (e.g. a lone accent
        # stripped by an uncased normalizer). Keep at least one byte so the patch axis
        # stays aligned with the token axis.
        if end_byte <= start_byte:
            end_byte = min(start_byte + 1, len(data))
        if start_byte >= len(data) or end_byte <= start_byte:
            lengths.append(0)
            continue

        chunk = data[start_byte:end_byte]
        byte_ids.extend(chunk)
        lengths.append(len(chunk))
        boundaries.extend([0] * len(chunk))
        boundaries[-1] = 1
        n_absorbed += char_start_b - start_byte
        cursor = end_byte
        last_real_token = len(lengths) - 1

    # Trailing gap: bytes no token covered.  They join the last content token, and they
    # are inserted *before* any trailing pseudo-byte.  Appending them after the separator
    # put [SEP] in the middle of the stream, which silently broke the bracketing the model
    # was trained with: the byte that terminates the sequence stopped terminating it.
    if cursor < len(data) and lengths:
        tail = data[cursor:]
        insert_at = len(byte_ids)
        while insert_at > 0 and byte_ids[insert_at - 1] >= 256:
            insert_at -= 1

        byte_ids[insert_at:insert_at] = list(tail)
        boundaries[insert_at:insert_at] = [0] * len(tail)
        boundaries[insert_at + len(tail) - 1] = 1
        if insert_at > 0:
            # The byte the tail continues is no longer that token's last byte.
            boundaries[insert_at - 1] = 0

        if last_real_token is None:
            lengths.append(len(tail))
        else:
            lengths[last_real_token] += len(tail)
        n_absorbed += len(tail)

    if len(boundaries) != len(byte_ids) or sum(lengths) != len(byte_ids):
        raise AssertionError(
            f"alignment length mismatch: {len(boundaries)} boundaries vs {len(byte_ids)} bytes"
        )

    return Alignment(
        input_ids=tuple(token_ids),
        byte_ids=tuple(byte_ids),
        lengths=tuple(lengths),
        boundaries=tuple(boundaries),
        covered=n_absorbed,
    )


def bracket_pseudo_bytes(special_token_ids: Mapping[int, int]) -> dict[int, int]:
    """The pseudo-bytes that bracket the stream, and only those.

    ``discover_special_bytes`` also maps ``[UNK]`` and any added special token, and those
    must **not** be emitted into the byte stream. An unknown token stands for a character
    the vocabulary cannot represent; emitting a pseudo-byte for it injects a value that is
    not a byte of the text, so the training stream and the text disagree at exactly the
    positions the tokenizer struggled with. The pseudo-byte values stay reserved -- the
    byte table's size does not change and old checkpoints still load -- but only the class
    and separator pseudo-bytes are ever emitted (ADR 0011).
    """
    ordered = list(special_token_ids.items())[:2]
    return dict(ordered)


def align_text(
    text: str,
    tokenizer,
    *,
    special_token_ids: Mapping[int, int] | None = None,
    add_special_tokens: bool = True,
    strategy: str | None = None,
) -> Alignment:
    """Tokenize ``text`` and align it, choosing a route that fits the tokenizer family.

    Which route is authoritative depends on the family, and getting this wrong
    silently corrupts the byte stream:

    * **Byte-level BPE** -- concatenate per-token byte strings. Offsets are *not*
      trustworthy: for ``"日本語のテキスト"`` the fast tokenizer reports overlapping
      duplicate spans ``(2,3), (2,3)`` for three consecutive tokens, because byte-level
      offsets are recovered by counting characters covered by the decoded token and
      break on multi-byte characters. Following the offsets duplicates characters.
      The byte stream is that of the tokenizer's NFC-normalised text, which is what the
      source model actually saw.
    * **Everything else** (WordPiece, SentencePiece) -- use offsets, absorbing the
      bytes that no token covers.

    ``strategy`` overrides detection; it defaults to the resolved vocabulary's family.
    """
    if strategy is None:
        from .vocab import resolve_token_bytes

        strategy = resolve_token_bytes(tokenizer).strategy

    if special_token_ids is None:
        special_token_ids = discover_special_bytes(tokenizer)

    # Only the bracketing pseudo-bytes are emitted. See `bracket_pseudo_bytes`.
    emitted = bracket_pseudo_bytes(special_token_ids)

    if strategy == "bytelevel":
        from .vocab import resolve_token_bytes

        resolution = resolve_token_bytes(tokenizer)
        token_ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        parts: list[bytes | int] = []
        for token_id in token_ids:
            if token_id in emitted:
                parts.append(emitted[token_id])
            else:
                parts.append(resolution.token_bytes[token_id])
        return align(token_ids, parts)

    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)

    # For WordPiece (and other families that apply a normalizer), the offsets are
    # computed on the normalizer's output, not the original text.  We must also
    # use the normalizer's output as the byte-source, otherwise the byte stream
    # will disagree with the offsets (e.g. "Byte" vs "byte" differs by one byte).
    normalizer = getattr(getattr(tokenizer, "backend_tokenizer", None), "normalizer", None)
    normalised_text = (
        normalizer.normalize_str(text) if normalizer is not None else text
    )
    return align_from_offsets(
        normalised_text, encoded["input_ids"], encoded["offset_mapping"],
        special_token_ids=emitted,
    )


def discover_special_bytes(tokenizer) -> dict[int, int]:
    """Map special token ids to pseudo-byte values, deterministically.

    The mapping only has to be stable and injective, not equal to any particular
    convention, because the pseudo-bytes exist solely to hold the byte axis in sync
    with the token axis.
    """
    attrs = ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id")
    out: dict[int, int] = {}
    for attr in attrs:
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            out[value] = 256 + len(out)
    # Any tokenizer-declared special token not covered by the canonical four.
    from .vocab import added_tokens

    for token_id, _, special in added_tokens(tokenizer):
        if special and token_id not in out:
            out[token_id] = 256 + len(out)
    return out


def align(token_ids: Sequence[int], parts: Sequence[bytes | int]) -> Alignment:
    """Align from per-token parts, where an ``int`` denotes a pseudo-byte.

    Pseudo-bytes live outside the 0..255 range, so they cannot be carried inside a
    ``bytes`` object; they are passed as plain ints instead.
    """
    byte_ids: list[int] = []
    lengths: list[int] = []
    boundaries: list[int] = []
    for token_id, part in zip(token_ids, parts):
        if isinstance(part, int):
            byte_ids.append(part)
            lengths.append(1)
            boundaries.append(1)
            continue
        byte_ids.extend(part)
        lengths.append(len(part))
        boundaries.extend([0] * len(part))
        if len(part) > 0:
            boundaries[-1] = 1
    if len(boundaries) != len(byte_ids) or sum(lengths) != len(byte_ids):
        raise AssertionError("alignment length mismatch")
    return Alignment(tuple(token_ids), tuple(byte_ids), tuple(lengths), tuple(boundaries))
