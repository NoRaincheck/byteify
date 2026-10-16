"""Byteification losses, following the paper's equations.

Stage 1 (subword-to-byte distillation; global model frozen)::

    L_stage1 = L_B + L_E + L_emb

Stage 2 (end-to-end; everything trainable)::

    L_stage2 = L_B + L_CE

Terms, and what each is for
---------------------------
``L_B``  Boundary BCE against ``B_subword`` (paper's ``L_B``, lambda_B = 4). Teaches
         the learned boundary predictor to reproduce the source tokenizer's decisions.
         The paper reports this reaches >99% accuracy quickly, and we reproduce that.

``L_E``  The paper's stitching loss::

             Y_E    = M_:n(T_subword(x))
             Yhat_E = M_:n(Pool(E(T(x)), B_subword(x)))
             L_E    = || Y_E - Yhat_E ||^2

         Compared at depth ``n`` inside the frozen model rather than at its input,
         because similar input vectors need not behave similarly in later layers.
         The paper uses n = 4; at 7 layers we stitch at full depth instead (see
         RESULTS.md).

``L_emb``  **Encoder-specific replacement for ``L_{D,Distill}``.** For an LM, Stage 1's
         output-side loss recovers next-token likelihood. An embedder has no output
         distribution, so "recover the source model's behaviour" has to be chosen. We
         align the *sentence embedding* -- batch-centred cosine after the model's output
         norm -- which is the representation an embedding model is actually judged on.

         Centring is essential and not optional: mean-pooled embeddings share one
         dominant direction, so raw cosine sits near 1.0 almost regardless of content.
         An uncentred version of this loss reports ~0.99 while the two models agree on
         nearest neighbours no better than chance.

``L_CE``  Stage 2's byte-level cross-entropy. Bolmo keeps next-byte CE; the encoder
         analogue is **masked** byte prediction, since there is no next byte to predict
         from a bidirectional model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Stage1Losses:
    total: Tensor
    boundary: Tensor
    stitching: Tensor
    embedding: Tensor


def boundary_loss(
    boundary_logits: Tensor,
    boundaries: Tensor,
    byte_mask: Tensor,
    *,
    pos_weight: float | None = None,
) -> Tensor:
    """``L_B``: binary cross-entropy against ``B_subword`` on real bytes only.

    ``pos_weight`` optionally upweights the rare positive class. Boundaries are ~1 in 4.3
    bytes, so the class is imbalanced but not severely; the paper uses plain BCE.
    """
    valid = byte_mask.bool()
    if valid.sum() == 0:
        return boundary_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        boundary_logits[valid],
        boundaries[valid].float(),
        pos_weight=pos_weight,
    )


def stitching_loss(student: Tensor, teacher: Tensor, patch_mask: Tensor) -> Tensor:
    """``L_E``: masked mean-squared error between representations at depth ``n``."""
    diff = (student.float() - teacher.float()) ** 2
    per_pos = diff.mean(dim=-1)
    m = patch_mask.to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum().clamp_min(1.0)


def pool(
    hidden: Tensor,
    mask: Tensor,
    final_norm: torch.nn.Module | None = None,
) -> Tensor:
    """Mean-pool over valid positions, applying the model's output norm first."""
    if final_norm is not None:
        hidden = final_norm(hidden)
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)


def embedding_loss(
    student_final: Tensor,
    teacher_final: Tensor,
    student_mask: Tensor,
    teacher_mask: Tensor,
    final_norm: torch.nn.Module | None = None,
) -> Tensor:
    """Encoder analogue of ``L_{D,Distill}``: batch-centred cosine on pooled embeddings."""
    s = pool(student_final, student_mask, final_norm)
    t = pool(teacher_final, teacher_mask, final_norm)
    s = s - s.mean(0, keepdim=True)
    t = t - t.mean(0, keepdim=True)
    return 1.0 - F.cosine_similarity(s, t, dim=-1).mean()


def masked_byte_loss(
    logits: Tensor,
    target_bytes: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """``L_CE``: cross-entropy on masked bytes.

    ``logits`` covers the byte alphabet only (real bytes 0..255); pseudo-bytes standing
    in for special tokens are excluded from ``loss_mask`` by the caller and never need
    to be predicted.
    """
    valid = loss_mask.bool()
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), target_bytes[valid])


def sample_mlm_mask(
    byte_ids: Tensor,
    byte_mask: Tensor,
    *,
    mlm_prob: float = 0.15,
    special_prob: float = 0.1,
    random_prob: float = 0.1,
    n_bytes: int = 256,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a BERT-style byte masking pattern.

    Returns ``(inputs, loss_mask)``. ``inputs`` is the corrupted byte stream that the
    student sees; ``loss_mask`` marks positions scored against the *original* bytes.

    Masking happens at the byte level, not the token level. That is the point of the
    architecture: the model never has a token boundary to hide behind.
    """
    device = byte_ids.device
    probs = torch.rand(byte_ids.shape, device=device, generator=generator)
    selected = (probs < mlm_prob) & byte_mask.bool()
    # Never mask pseudo-bytes: they have no recoverable identity.
    selected = selected & (byte_ids < n_bytes)

    replace_mask = selected & (torch.rand(byte_ids.shape, device=device, generator=generator) < special_prob)
    random_mask = selected & ~replace_mask & (
        torch.rand(byte_ids.shape, device=device, generator=generator) < random_prob
    )

    inputs = byte_ids.clone()
    inputs[replace_mask] = torch.randint(
        0, n_bytes, (int(replace_mask.sum()),), device=device, generator=generator
    )
    if random_mask.any():
        inputs[random_mask] = torch.randint(
            0, n_bytes, (int(random_mask.sum()),), device=device, generator=generator
        )
    return inputs, selected


@dataclass
class Stage1Weights:
    """Paper defaults: lambda_B = 4, lambda_E = 1."""

    boundary: float = 4.0
    stitching: float = 1.0
    embedding: float = 1.0


def stage1_total(losses: dict[str, Tensor], weights: Stage1Weights) -> Stage1Losses:
    total = (
        weights.boundary * losses["boundary"]
        + weights.stitching * losses["stitching"]
        + weights.embedding * losses["embedding"]
    )
    return Stage1Losses(
        total=total,
        boundary=losses["boundary"].detach(),
        stitching=losses["stitching"].detach(),
        embedding=losses["embedding"].detach(),
    )