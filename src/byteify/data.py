"""Corpus loading and length-grouped batching for byte-level training.

Byte-level training has a shape problem subword training does not: one document
produces ~4.3x more positions, and raw web documents vary in length by orders of
magnitude. Padding a batch of unrelated documents to the longest wastes most of the
compute, so batches are formed from documents of similar length.

The sampler is deliberately simple: shuffle, cut into megabatches, sort each megabatch
by length, emit equal-size batches, shuffle the batch order. This is enough to keep
padding waste low without needing a full sortable sampler.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .stitching import ByteBatch, encode_texts


def iter_texts(
    paths: Sequence[Path | str],
    *,
    min_chars: int = 0,
    max_chars: int | None = None,
    limit: int | None = None,
) -> Iterator[str]:
    """Yield text from ``.jsonl`` (``{"text": ...}``) or plain-text shards."""
    n = 0
    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get("text", "")
                    if len(text) < min_chars:
                        continue
                    yield text[:max_chars] if max_chars else text
                    n += 1
                    if limit and n >= limit:
                        return
        else:
            with path.open(encoding="utf-8", errors="replace") as f:
                buffer: list[str] = []
                for line in f:
                    buffer.append(line.strip())
                    if len(" ".join(buffer)) >= (max_chars or 4000):
                        text = " ".join(buffer)
                        if len(text) >= min_chars:
                            yield text
                            n += 1
                            if limit and n >= limit:
                                return
                        buffer = []


@dataclass
class BatchSampler:
    """Length-grouped batches over an in-memory list of documents."""

    texts: list[str]
    tokenizer: object
    batch_size: int = 8
    max_bytes: int | None = None
    add_special_tokens: bool = True
    seed: int = 0
    megabatch: int = 50

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # Cache byte lengths once: tokenizing here would defeat the batching.
        self._lengths = [len(t.encode("utf-8")) for t in self.texts]

    def __len__(self) -> int:
        return len(self.texts) // self.batch_size

    def epoch(self) -> Iterator[ByteBatch]:
        order = list(range(len(self.texts)))
        self._rng.shuffle(order)
        batches: list[list[int]] = []
        for start in range(0, len(order), self.batch_size * self.megabatch):
            window = order[start : start + self.batch_size * self.megabatch]
            window.sort(key=lambda i: self._lengths[i])
            for b in range(0, len(window), self.batch_size):
                chunk = window[b : b + self.batch_size]
                if len(chunk) == self.batch_size:
                    batches.append(chunk)
        self._rng.shuffle(batches)

        for batch in batches:
            texts = [self._prepare(self.texts[i]) for i in batch]
            yield encode_texts(
                texts,
                self.tokenizer,
                add_special_tokens=self.add_special_tokens,
            )

    def _prepare(self, text: str) -> str:
        return truncate_to_bytes(text, self.max_bytes)


def truncate_to_bytes(text: str, max_bytes: int | None) -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes, on a character boundary.

    One rule, so that a training batch and an evaluation batch are built from identically
    truncated text. Truncating the training sampler but not the evaluation path meant
    ``--max-bytes`` silently did nothing to evaluation, and the byte axis there was set by
    whichever document happened to be longest.
    """
    if max_bytes is None:
        return text
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    """Pad the last dimension up to a multiple, returning ``(tensor, pad_len)``."""
    length = tensor.shape[-1]
    remainder = (-length) % multiple
    if remainder == 0:
        return tensor, 0
    pad_shape = list(tensor.shape)
    pad_shape[-1] = remainder
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=-1), remainder