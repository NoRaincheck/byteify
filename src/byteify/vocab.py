"""Resolving a tokenizer's vocabulary to byte strings, across tokenizer families.

Byteification needs to know, for every vocabulary entry, which byte string it denotes:

* the boundary supervision ``B_subword`` is the token sequence projected onto bytes, and
* the ``T_SubwordSuffix`` residual needs a byte trie over the vocabulary.

Only some tokenizers state this directly. This module resolves it for the families
that actually appear on the Hub, and reports what it could not resolve instead of
guessing.

Supported
---------
``bytelevel``  Byte-level BPE (OLMo-2 / ModernBERT / ettin / GPT-2). Every entry is a
              string over the 256-symbol ByteLevel alphabet, which maps bijectively
              back to bytes.
``wordpiece``  WordPiece (BERT / MiniLM / E5 / BGE). Entries are literal strings;
              a leading ``##`` marks a word continuation and is not part of the text.
``sentencepiece``  Unigram/BPE with the SentencePiece marker (XLM-R / ALBERT). ``▁``
              stands for a space.
``byte_fallback``  BPE with byte-fallback tokens spelled ``<0x41>`` (XLM-R legacy,
              ByT5). Those entries denote exactly one byte.

Two properties are *not* recoverable and are handled explicitly:

* **Normalizers that destroy information.** ``bert-base-uncased`` lowercases and
  strips accents, so ``naïve`` is tokenised as ``naive``. The token's byte string
  genuinely is not the bytes of the source text; the trie will simply fail to match
  there and fall back to a shorter suffix. This is a property of the teacher, not a bug
  here -- but it means the suffix residual is weaker for such tokenizers.
* **Unmappable entries** (``[UNK]``, control tokens, normalizer artefacts) simply have
  no byte string. They are excluded from the trie, so those positions get no suffix
  contribution and the byte embedding carries them alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .alignment import SPECIAL_TOKENS, bytes_to_unicode

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_SPM_MARKER = "▁"


@dataclass
class VocabByteResolution:
    """token id -> byte string, plus bookkeeping about what was not resolvable."""

    strategy: str
    token_bytes: tuple[bytes, ...]
    unmapped: tuple[int, ...] = field(default=())

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    @property
    def coverage(self) -> float:
        mapped = sum(1 for b in self.token_bytes if b)
        return mapped / max(len(self.token_bytes), 1)

    def describe(self) -> str:
        return (
            f"{self.strategy}: {self.vocab_size} entries, "
            f"{self.coverage * 100:.1f}% with bytes, {len(self.unmapped)} unmapped"
        )


def _byte_level_table(tokenizer) -> dict[int, bytes]:
    char_to_byte = {c: b for b, c in bytes_to_unicode().items()}
    table: dict[int, bytes] = {}
    for surface, token_id in tokenizer.get_vocab().items():
        if any(c not in char_to_byte for c in surface):
            continue
        table[token_id] = bytes(char_to_byte[c] for c in surface)
    return table


def _detect_strategy(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre = getattr(backend, "pre_tokenizer", None) if backend is not None else None
    if type(pre).__name__ == "ByteLevel":
        return "bytelevel"
    model = getattr(backend, "model", None) if backend is not None else None
    if type(model).__name__ == "WordPiece":
        return "wordpiece"
    if type(model).__name__ in {"BPE", "Unigram"}:
        vocab = tokenizer.get_vocab()
        if any(_BYTE_FALLBACK.match(s) for s in vocab):
            return "byte_fallback"
        if any(_SPM_MARKER in s for s in vocab):
            return "sentencepiece"
        return "bytelevel"
    return "literal"


_RESOLUTION_CACHE: dict[int, VocabByteResolution] = {}
_SPECIAL_CACHE: dict[int, set[int]] = {}


def resolve_token_bytes(tokenizer, *, use_cache: bool = True) -> VocabByteResolution:
    """Resolve every vocabulary entry to its byte string where possible.

    Cached per tokenizer: building this walks the whole vocabulary, and the alignment
    path calls it once per training example, which dominated preprocessing cost.
    """
    key = id(tokenizer)
    if use_cache and key in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[key]
    result = _resolve_token_bytes(tokenizer)
    if use_cache:
        _RESOLUTION_CACHE[key] = result
    return result


def _resolve_token_bytes(tokenizer) -> VocabByteResolution:
    strategy = _detect_strategy(tokenizer)
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    table: dict[int, bytes] = {}

    if strategy == "bytelevel":
        table = _byte_level_table(tokenizer)
    elif strategy == "wordpiece":
        for surface, token_id in vocab.items():
            if surface in SPECIAL_TOKENS or not surface:
                continue
            body = surface.removeprefix("##")
            table[token_id] = body.encode("utf-8")
    elif strategy == "sentencepiece":
        for surface, token_id in vocab.items():
            if surface in SPECIAL_TOKENS or not surface:
                continue
            table[token_id] = surface.replace(_SPM_MARKER, " ").encode("utf-8")
    elif strategy == "byte_fallback":
        for surface, token_id in vocab.items():
            if surface in SPECIAL_TOKENS or not surface:
                continue
            m = _BYTE_FALLBACK.match(surface)
            if m:
                table[token_id] = bytes([int(m.group(1), 16)])
            elif _SPM_MARKER in surface:
                table[token_id] = surface.replace(_SPM_MARKER, " ").encode("utf-8")
            else:
                table[token_id] = surface.encode("utf-8")
    else:  # literal
        for surface, token_id in vocab.items():
            if surface in SPECIAL_TOKENS or not surface:
                continue
            table[token_id] = surface.encode("utf-8")

    # Added tokens are matched as literal strings, regardless of family.
    for token_id, content, _ in added_tokens(tokenizer):
        if content in SPECIAL_TOKENS or not content:
            continue
        table[token_id] = content.encode("utf-8")

    token_bytes = [b""] * size
    for token_id, seq in table.items():
        if 0 <= token_id < size:
            token_bytes[token_id] = seq
    unmapped = tuple(i for i, b in enumerate(token_bytes) if not b and i not in _special_ids(tokenizer))
    return VocabByteResolution(strategy, tuple(token_bytes), unmapped)


def added_tokens(tokenizer) -> list[tuple[int, str, bool]]:
    """(id, content, special) for every added token, across transformers versions.

    ``AddedToken`` has varied across releases -- sometimes a mapping, sometimes an
    object, with inconsistent ``.id`` support -- so read defensively and prefer the
    dict key, which is always present when the mapping is keyed by id.
    """
    out: list[tuple[int, str, bool]] = []
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    items = decoder.items() if hasattr(decoder, "items") else enumerate(decoder)
    for key, entry in items:
        if isinstance(entry, dict):
            content = entry.get("content", "")
            special = bool(entry.get("special", False))
            token_id = int(entry.get("id", key))
        else:
            content = getattr(entry, "content", "")
            special = bool(getattr(entry, "special", False))
            token_id = int(getattr(entry, "id", key))
        out.append((token_id, content, special))
    return out


def _special_ids(tokenizer) -> set[int]:
    key = id(tokenizer)
    if key in _SPECIAL_CACHE:
        return _SPECIAL_CACHE[key]
    ids = {token_id for token_id, content, _ in added_tokens(tokenizer) if content in SPECIAL_TOKENS}
    for token_id, _, special in added_tokens(tokenizer):
        if special:
            ids.add(token_id)
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    _SPECIAL_CACHE[key] = ids
    return ids


def build_suffix_trie(tokenizer):
    """Byte trie over the resolvable vocabulary, for ``T_SubwordSuffix``.

    Returns ``(trie, resolution)``. Entries that could not be resolved are absent from
    the trie by construction; positions they would have covered resolve to a shorter
    suffix or to nothing.
    """
    from .expand import ByteTrie

    resolution = resolve_token_bytes(tokenizer)
    specials = _special_ids(tokenizer)
    return ByteTrie(
        {i: b for i, b in enumerate(resolution.token_bytes) if b and i not in specials}
    ), resolution