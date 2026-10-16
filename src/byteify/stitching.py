"""Turning texts into padded byte / patch batches, and the Stage 1 stitching loss.

Stage 1 of byteification trains the local encoder to reproduce the source model's
representations. Following the paper we compare *inside* the frozen global model
rather than directly at its input, because similar input vectors need not behave
similarly in later layers (model stitching).::

    Y_E     = M_:n(T_subword(x))
    Yhat_E  = M_:n(Pool(E(T(x)), B_subword(x)))
    L_E     = || Y_E - Yhat_E ||

With n = 0 this degenerates to matching the subword embedding matrix, which is what
Hwang et al. do; the paper found n = 4 substantially better for a small compute cost.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .alignment import align_text, discover_special_bytes
from .expand import expand_byte_ids

_SHARED: weakref.WeakKeyDictionary[object, tuple] = weakref.WeakKeyDictionary()


def tokenizer_tables(tokenizer):
    """Cache per-tokenizer tables: special pseudo-bytes and the suffix trie.

    Keyed on the tokenizer object rather than ``id()``: the address is recycled once the
    tokenizer is collected, and a process that byteifies several models can otherwise be
    handed the previous model's trie.
    """
    hit = _SHARED.get(tokenizer)
    if hit is not None:
        return hit
    from .vocab import build_suffix_trie

    trie, resolution = build_suffix_trie(tokenizer)
    tables = (discover_special_bytes(tokenizer), trie, resolution)
    _SHARED[tokenizer] = tables
    return tables


@dataclass
class ByteBatch:
    """A padded batch, carrying both the subword view and the byte view."""

    input_ids: Tensor  # [B, T] subword tokens
    attention_mask: Tensor  # [B, T]
    byte_ids: Tensor  # [B, N] byte values / pseudo-bytes
    expanded_ids: Tensor  # [B, N] longest-suffix subword id per byte
    byte_mask: Tensor  # [B, N]
    boundaries: Tensor  # [B, N] B_subword supervision
    lengths: Tensor  # [B, T] bytes contributed by each token (0 for dropped specials)

    def to(self, device: str | torch.device) -> ByteBatch:
        return ByteBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]


def encode_texts(
    texts: Sequence[str],
    tokenizer,
    *,
    add_special_tokens: bool = True,
    device: str | torch.device = "cpu",
) -> ByteBatch:
    """Tokenize, align, expand, and pad a list of texts.

    Works for any tokenizer family: alignment comes from offsets, and the suffix trie
    from the resolved vocabulary. See :mod:`byteify.alignment` for why offsets rather
    than token concatenation.
    """
    special_bytes, trie, _ = tokenizer_tables(tokenizer)
    pseudo_to_token = {pseudo: tid for tid, pseudo in special_bytes.items()}

    subword_rows: list[list[int]] = []
    byte_rows: list[list[int]] = []
    expanded_rows: list[list[int]] = []
    boundary_rows: list[list[int]] = []
    length_rows: list[list[int]] = []

    for text in texts:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
        a = align_text(
            text,
            tokenizer,
            special_token_ids=special_bytes,
            add_special_tokens=add_special_tokens,
        )
        assert list(encoded["input_ids"]) == list(a.input_ids), "alignment disagrees with tokenizer"

        byte_ids = list(a.byte_ids)
        expanded = expand_byte_ids(byte_ids, trie, pseudo_byte_to_token=pseudo_to_token)
        assert len(expanded) == len(byte_ids)

        subword_rows.append(list(a.input_ids))
        byte_rows.append(byte_ids)
        expanded_rows.append(expanded)
        boundary_rows.append(list(a.boundaries))
        length_rows.append(list(a.lengths))

    def pad(rows: list[list[int]], value: int) -> Tensor:
        width = max((len(r) for r in rows), default=1)
        out = torch.full((len(rows), width), value, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return out

    T = max((len(r) for r in subword_rows), default=1)

    input_ids = torch.full((len(texts), T), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(texts), T), dtype=torch.long)
    for i, r in enumerate(subword_rows):
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        attention_mask[i, : len(r)] = 1

    byte_ids = pad(byte_rows, 0)
    expanded_ids = pad(expanded_rows, -1)
    boundaries = pad(boundary_rows, 0)
    lengths = pad(length_rows, 0)
    byte_mask = pad([[1] * len(r) for r in byte_rows], 0)

    # Padding must not close a patch: only real final bytes may carry a boundary.
    boundaries = boundaries * byte_mask

    return ByteBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        byte_ids=byte_ids,
        expanded_ids=expanded_ids,
        byte_mask=byte_mask,
        boundaries=boundaries,
        lengths=lengths,
    ).to(device)


def masked_l2(student: Tensor, teacher: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over valid positions only."""
    diff = (student.float() - teacher.float()) ** 2
    per_pos = diff.mean(dim=-1)  # [B, P]
    mask = mask.to(per_pos.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (per_pos * mask).sum() / denom


@torch.no_grad()
def teacher_states(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    n_layers: Sequence[int],
) -> dict[int, Tensor]:
    """The subword model's representations at each requested depth.

    ``n_layers = 0`` returns the post-embedding-norm representation, i.e. the point
    ``inputs_embeds`` feeds into when the student is run through the same model.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = out.hidden_states
    result = {0: model.embeddings(input_ids=input_ids)}
    for n in n_layers:
        if n != 0:
            result[n] = hidden[n]
    return result


def cos_sim(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    """Mean cosine similarity over valid positions."""
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    sim = (a * b).sum(-1)
    m = mask.to(sim.dtype)
    return (sim * m).sum() / m.sum().clamp_min(1.0)