"""One rule for turning text into a byte stream, shared by training and inference.

ADR 0011. The rule is a value: it is declared, recorded in the manifest, and applied
identically on both paths. When training and inference each had their own, they agreed
on every sample in the test suite and still disagreed in production, because training
went through the tokenizer's offsets and inference truncated differently.

Three rules, by what the source tokenizer does to text before tokenising it:

``nfc``
    Byte-level BPE. The source model saw NFC-normalised text, so the byte stream is NFC
    and nothing else. Normalisation is ``unicodedata.normalize`` -- no tokenizer call --
    and is exact across all 158,867 codepoints.

``raw-utf8``
    A tokenizer with no text normaliser. The bytes are the text's bytes.

``normalised``
    A tokenizer that rewrites text (lowercasing, accent stripping, punctuation
    splitting). Reproducing its output *is* reproducing the tokenizer, so this rule
    cannot be applied without the tokenizer's normaliser. It is declared rather than
    hidden, because an artifact built for one of these is honestly not tokenizer-free, and
    the manifest is where a practitioner reads that.

Pseudo-bytes bracket the stream at both ends on every path. That is forced rather than
chosen: the encoder was trained with ``[CLS]``/``[SEP]``-shaped pseudo-bytes at positions 0
and N-1 and has never seen the stream without them.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

RuleName = Literal["nfc", "raw-utf8", "normalised"]

#: Families whose byte stream needs no tokenizer.
TOKENIZER_FREE_RULES: frozenset[str] = frozenset({"nfc", "raw-utf8"})


@dataclass(frozen=True, slots=True)
class ByteStreamRule:
    """A declared byte-stream rule, plus the pseudo-bytes that bracket the stream."""

    rule: RuleName
    family: str
    n_byte_values: int
    open_pseudo: int = -1
    close_pseudo: int = -1
    pseudo_bytes: dict[int, int] = field(default_factory=dict)

    @property
    def tokenizer_free(self) -> bool:
        """Whether this rule can be applied with no tokenizer present."""
        return self.rule in TOKENIZER_FREE_RULES

    def content(self, text: str) -> bytes:
        """The text's bytes under this rule, without the bracketing pseudo-bytes."""
        if self.rule == "nfc":
            return unicodedata.normalize("NFC", text).encode("utf-8")
        if self.rule == "raw-utf8":
            return text.encode("utf-8")
        raise NotImplementedError(
            "rule 'normalised' reproduces a tokenizer's own text rewriting and cannot be "
            "applied without it; byteify cannot build inference inputs for this artifact "
            "without the source tokenizer's normaliser"
        )

    def build(self, text: str, max_bytes: int) -> list[int]:
        """Text -> byte ids, bracketed, no longer than ``max_bytes``.

        ``max_bytes`` bounds the *whole* axis, pseudo-bytes included, so that a serving
        path sizing hardware from the artifact's declared byte axis gets what it expects.
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be at least 1; got {max_bytes}")
        overhead = (self.open_pseudo >= 0) + (self.close_pseudo >= 0)
        budget = max(0, max_bytes - overhead)

        raw = self.content(text)[:budget]
        # Truncation lands mid-character otherwise, and the surviving half of a
        # multi-byte character is not valid UTF-8 and not what the source model saw.
        text_so_far = raw.decode("utf-8", errors="ignore")
        stream = list(text_so_far.encode("utf-8"))

        if self.open_pseudo >= 0:
            stream.insert(0, self.open_pseudo)
        if self.close_pseudo >= 0:
            stream.append(self.close_pseudo)
        return stream

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "family": self.family,
            "n_byte_values": self.n_byte_values,
            "open_pseudo": self.open_pseudo,
            "close_pseudo": self.close_pseudo,
            "pseudo_bytes": {str(k): v for k, v in sorted(self.pseudo_bytes.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ByteStreamRule:
        return cls(
            rule=str(d["rule"]),  # type: ignore[arg-type]
            family=str(d.get("family", "")),
            n_byte_values=int(d.get("n_byte_values", 256)),
            open_pseudo=int(d.get("open_pseudo", -1)),
            close_pseudo=int(d.get("close_pseudo", -1)),
            pseudo_bytes={int(k): int(v) for k, v in (d.get("pseudo_bytes") or {}).items()},
        )


def rule_for_family(
    family: str,
    *,
    n_byte_values: int = 256,
    special_tokens: dict[int, int] | None = None,
) -> ByteStreamRule:
    """The declared rule for a tokenizer family.

    ``special_tokens`` is what ``discover_special_bytes`` returns: token id to pseudo byte,
    in the order the tokenizer's attributes were probed.
    """
    special_tokens = special_tokens or {}
    special_tokens = special_tokens or {}
    pseudo_to_token = {pseudo: token_id for token_id, pseudo in special_tokens.items()}
    # `discover_special_bytes` preserves the order it probed attributes in -- class,
    # separator, pad, mask -- so the first two values it assigned are the ones that frame
    # the byte axis, whatever the token ids happen to be.
    #
    # Only those two are ever emitted. [UNK] and the rest keep their reserved values, so
    # the byte table's size does not change, but an unknown token stands for text the
    # vocabulary cannot represent: emitting a value for it would put a non-byte into the
    # stream at exactly the positions the tokenizer struggled.
    ordered = list(special_tokens.values())[:2]
    open_pseudo = ordered[0] if len(ordered) > 0 else -1
    close_pseudo = ordered[1] if len(ordered) > 1 else -1

    literal: RuleName = "nfc" if family == "bytelevel" else "raw-utf8"
    return ByteStreamRule(
        rule=literal,
        family=family,
        n_byte_values=n_byte_values,
        open_pseudo=open_pseudo,
        close_pseudo=close_pseudo,
        pseudo_bytes=pseudo_to_token,
    )