"""Fidelity: does the byteified model still behave like the model it came from?

One implementation, used by the Stage 1 trainer to report progress and by the export gate
to decide whether an artifact is allowed to exist. When those were separate the trainer
reported numbers the gate could not reproduce, and a gate that cannot reproduce the
trainer's numbers is a gate nobody trusts.

Every embedding metric here is **centred** (ADR 0005). Raw cosine between mean-pooled
embeddings reads 0.997 whether or not two models agree, because pooled embeddings are
strongly anisotropic and share one dominant direction. Centring makes the number respond
to the component that actually separates documents; the raw value is reported alongside
it so the difference stays visible.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def pool_final(h: Tensor, mask: Tensor, final_norm=None) -> Tensor:
    """Mean-pool over valid positions, matching how the source model pools."""
    if final_norm is not None:
        h = final_norm(h)
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)


def pearson(a: Tensor, b: Tensor) -> Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)


def _budgeted_patch_reps(embedder, batch):
    """Patch representations under the artifact's own budgeted patching.

    This is the gate's measurement, because budgeted patching is what the exported graph
    does: a fixed number of patches, chosen by confidence, with low-confidence slots
    masked. A fidelity number measured any other way would be a number about a model the
    practitioner is not going to deploy.
    """
    from .inference import select_budgeted_patches

    encoder = embedder.encoder
    h = encoder.local_encoder(encoder.embed(batch.byte_ids, batch.expanded_ids), batch.byte_mask)
    probs = torch.sigmoid(encoder.boundary_predictor(h))
    # Clamp to this batch's byte axis, exactly as the serving path does in `_embedder_core`.
    # A batch of short texts pads to fewer bytes than the patch axis, and asking for more
    # patches than there are byte positions raised rather than filled what it could: the
    # contrastive stage died on any corpus of short texts, including this repository's own
    # synthetic one.
    patches = select_budgeted_patches(probs, batch.byte_mask,
                                      min(embedder.n_patches, batch.byte_mask.shape[1]),
                                      embedder.threshold)
    idx = patches.end_indices.unsqueeze(-1).expand(-1, -1, h.shape[-1])
    reps = h.gather(1, idx) * patches.mask.unsqueeze(-1).to(h.dtype)
    return reps, patches.mask, probs


def _oracle_patch_reps(encoder, batch):
    """Patch representations under the oracle subword boundaries.

    This is the trainer's measurement, because Stage 1's losses pool with the true
    boundaries so the student's patch axis stays aligned with the teacher's token axis.
    Measuring the trainer's own rule keeps the reported number honest about what was
    optimised, rather than about what will be deployed.
    """
    out = encoder(
        batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
    )
    return out.patch_reps, out.patch_mask, torch.sigmoid(out.boundary_logits)


@torch.no_grad()
def measure_fidelity(
    patches_for,
    source,
    batches,
    *,
    restore: object | None = None,
) -> dict[str, float]:
    """Measure agreement between the byteified model and the frozen source model.

    ``patches_for(batch) -> (patch_reps, patch_mask, boundary_probs)`` decides how patches
    are formed. Pass :func:`_budgeted_patch_reps` for the gate and
    :func:`_oracle_patch_reps` for the trainer; they are different measurements and
    conflating them is how a trainer ends up reporting a number the artifact does not
    reproduce.

    Reports:

    ``cos_emb_c``    centred cosine of pooled sentence embeddings
    ``cos_emb_raw``  the same without centring, so the gap stays visible (ADR 0005)
    ``sim_corr``     correlation between the two pairwise similarity matrices
    ``nn_agree``     nearest-neighbour agreement
    ``boundary_acc`` boundary-decision agreement
    ``cos@depth``    per-position cosine at the stitching depth

    **Centring is over the whole evaluation set, not over each batch.** Removing the
    dominant direction means subtracting the mean of the population, and a batch of four
    does not estimate a mean: the same checkpoint measured 0.885 through this function with
    batches of sixteen and 0.672 with batches of four, on the same held-out texts. A
    fidelity number that moves when the batching moves is measuring the batch, so the
    centring happens once, after every batch has been pooled. ``cos_emb_raw`` is unaffected
    by this and is reported precisely so the two can be compared.
    """
    s_pooled: list[Tensor] = []
    t_pooled: list[Tensor] = []
    cos_by_depth: dict[int, list[float]] = {}
    b_correct = 0.0
    b_total = 0
    depth = source.n_layers

    for batch in batches:
        patch_reps, patch_mask, probs = patches_for(batch)
        teacher = source.representation_at(batch.input_ids, batch.attention_mask, depth)
        student = source.forward_layers(patch_reps, patch_mask, n_layers=depth)

        s = pool_final(student, patch_mask, source.final_norm)
        t = pool_final(teacher, batch.attention_mask.bool(), source.final_norm)
        s_pooled.append(s.cpu())
        t_pooled.append(t.cpu())

        cos_by_depth.setdefault(depth, []).extend(
            F.cosine_similarity(student, teacher, dim=-1)[patch_mask].flatten().cpu().tolist()
            if student.shape[1] == teacher.shape[1]
            # Budgeted patching produces a fixed number of patches that do not correspond
            # one-to-one with the teacher's tokens, so per-position comparison is undefined
            # here. Pooled similarity and boundary agreement still are, and they are what
            # the gate declares. Reported as absent rather than as a misleading zero.
            else [float("nan")]
        )

        valid = batch.byte_mask.bool()
        b_correct += ((probs > 0.5)[valid] == batch.boundaries[valid].bool()).sum().item()
        b_total += batch.boundaries[valid].numel()

    s_raw = torch.cat(s_pooled)
    t_raw = torch.cat(t_pooled)
    # One centring for the whole evaluation set, so the metric does not depend on how the
    # texts were batched on the way in.
    s_all = F.normalize(s_raw - s_raw.mean(0, keepdim=True), dim=-1)
    t_all = F.normalize(t_raw - t_raw.mean(0, keepdim=True), dim=-1)
    n = s_all.shape[0]
    if n < 3:
        raise ValueError(
            f"fidelity needs at least 3 texts to correlate similarity matrices; got {n}"
        )
    upper = torch.triu_indices(n, n, offset=1)
    s_sim = s_all @ s_all.T
    t_sim = t_all @ t_all.T
    # The raw number, computed from the uncentred embeddings. Computing it from the
    # centred ones -- which is what this did -- makes it identical to ``cos_emb_c`` by
    # construction, and a metric that cannot show the anisotropy it exists to expose
    # cannot warn anyone about it.
    s_unit = F.normalize(s_raw, dim=-1)
    t_unit = F.normalize(t_raw, dim=-1)

    return {
        "cos_emb_c": (s_all * t_all).sum(-1).mean().item(),
        "cos_emb_raw": F.cosine_similarity(s_unit, t_unit, dim=-1).mean().item(),
        # How aligned the two models' dominant directions are. Reported next to the raw
        # cosine because "raw 0.99" is mostly this number, and an artifact whose dominant
        # direction matches its source's while its centred geometry does not is exactly the
        # failure mode centring exists to catch.
        "dominant_dir_cosine": (
            F.cosine_similarity(s_unit.mean(0, keepdim=True), t_unit.mean(0, keepdim=True))
            .squeeze()
            .item()
        ),
        "sim_corr": pearson(s_sim[upper[0], upper[1]], t_sim[upper[0], upper[1]]).item(),
        "nn_agree": (
            s_sim.fill_diagonal_(float("-inf")).argmax(-1)
            == t_sim.fill_diagonal_(float("-inf")).argmax(-1)
        ).float().mean().item(),
        "nn_chance": 1.0 / max(n - 1, 1),
        "boundary_acc": b_correct / max(b_total, 1),
        **{
            f"cos@{d}": (
                sum(v) / len(v) if not any(math.isnan(x) for x in v) else None
            )
            for d, v in cos_by_depth.items()
        },
        "eval_n": n,
    }