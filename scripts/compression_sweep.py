"""The paper's efficiency experiment: trading compression for fidelity.

A byteified model is not locked to a subword tokenizer's fixed patch set, so the
boundary supervision can be *thinned* -- adjacent patches fused -- to raise bytes per
patch and shorten the global model's sequence. The paper reports this as the mechanism
behind its Pareto result (Fig. 2), and contrasts it with a subword model, which would
need a larger vocabulary and pays a growing softmax cost.

For each target compression this script measures:

* **achieved bytes/patch** -- what the merging actually produced;
* **boundary accuracy** against the merged targets -- how well the predictor follows;
* **fidelity** to the source model -- similarity-matrix correlation and centred cosine,
  computed with merged boundaries, i.e. what the model would do at that compression.

    uv run python scripts/compression_sweep.py --init runs/stage1/stage1.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.boundary_merge import merge_batch
from byteify.data import BatchSampler, iter_texts
from byteify.source import HuggingFaceSourceEncoder, load_byteified_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--init", default="runs/stage1/stage1.pt")
    p.add_argument("--data", nargs="*", default=["data/corpus/fineweb-edu.jsonl"])
    p.add_argument("--targets", nargs="*", type=float, default=[4.3, 6.0, 8.0, 12.0, 16.0])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="runs/compression_sweep.json")
    return p.parse_args()


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)


@torch.no_grad()
def evaluate(encoder, source, batch, final_norm) -> dict[str, float]:
    out = encoder(
        batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
    )
    depth = source.n_layers
    student = source.forward_layers(out.patch_reps, out.patch_mask, n_layers=depth)
    teacher = source.representation_at(batch.input_ids, batch.attention_mask, depth)

    def pool(h, mask):
        if final_norm is not None:
            h = final_norm(h)
        m = mask.to(h.dtype).unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)

    s = pool(student, out.patch_mask)
    t = pool(teacher, batch.attention_mask.bool())
    s = torch.nn.functional.normalize(s - s.mean(0), dim=-1).cpu()
    t = torch.nn.functional.normalize(t - t.mean(0), dim=-1).cpu()
    n = s.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    s_sim, t_sim = s @ s.T, t @ t.T

    pred = (torch.sigmoid(out.boundary_logits) > 0.5)[batch.byte_mask.bool()]
    tgt = batch.boundaries[batch.byte_mask.bool()].bool()

    metrics = {
        "bytes_per_patch": batch.byte_mask.sum().item() / max(batch.boundaries.sum().item(), 1),
        "sim_corr": pearson(s_sim[iu[0], iu[1]], t_sim[iu[0], iu[1]]).item(),
        "cos_emb_c": (s * t).sum(-1).mean().item(),
        "nn_agree": (
            s_sim.fill_diagonal_(float("-inf")).argmax(-1)
            == t_sim.fill_diagonal_(float("-inf")).argmax(-1)
        ).float().mean().item(),
        "boundary_acc": (pred == tgt).float().mean().item(),
    }
    # Per-patch comparison against the token axis is only defined while patches and
    # tokens still correspond one-to-one. Merging boundaries breaks that alignment, so
    # this metric is reported as NaN above the source tokenizer's own ratio; the
    # sentence-level metrics above remain valid at any compression.
    if student.shape[1] == teacher.shape[1]:
        metrics["cos_depth"] = (
            torch.nn.functional.cosine_similarity(student, teacher, dim=-1)[out.patch_mask]
            .mean()
            .item()
        )
    else:
        metrics["cos_depth"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(args.model)).to(device).eval()

    if Path(args.init).exists():
        # The architecture comes out of the checkpoint, not out of constants here.
        encoder = load_byteified_encoder(args.init, source, tokenizer).to(device)
        print(f"loaded {args.init}")
    else:
        from byteify.source import build_byteified_encoder

        encoder = build_byteified_encoder(source, tokenizer).to(device)
        encoder.embed.load_suffix_from(source.embedding_matrix())
        print(f"WARNING: {args.init} not found; using a random local encoder")
    encoder.eval()

    texts = list(iter_texts(args.data, min_chars=64, limit=args.n_train))
    sampler = BatchSampler(
        texts, tokenizer, batch_size=args.batch_size, max_bytes=args.max_bytes, seed=0
    )
    base_batches = []
    for i, batch in enumerate(sampler.epoch()):
        if i >= args.batches:
            break
        base_batches.append(batch.to(device))
    n_eval = base_batches[0].byte_ids.shape[0] * len(base_batches)

    print(
        f"\nevaluating {len(base_batches)} batches ({n_eval} texts) at "
        f"each target compression\n"
    )
    header = f"{'target':>7} {'achieved':>9} {'bnd_acc':>8} {'sim_corr':>9} {'cos_emb_c':>10} {'nn_agree':>9} {'cos@L':>7}"
    print(header)
    print("-" * len(header))

    results = []
    # Baseline: no merging at all. This is the point where patches and tokens still
    # correspond one-to-one, so it is the only row where `cos@L` is defined.
    per_batch = [evaluate(encoder, source, b, source.final_norm) for b in base_batches]
    baseline = {
        k: sum(d[k] for d in per_batch) / len(per_batch) for k in per_batch[0]
    }
    row = {"target": float("nan"), "achieved": baseline["bytes_per_patch"], **baseline}
    results.append(row)
    print(
        f"{'none':>7} {baseline['bytes_per_patch']:>9.2f} {baseline['boundary_acc']:>8.4f} "
        f"{baseline['sim_corr']:>9.3f} {baseline['cos_emb_c']:>10.3f} "
        f"{baseline['nn_agree']:>9.3f} {baseline['cos_depth']:>7.3f}",
        flush=True,
    )

    for target in args.targets:
        merged = [merge_batch(b, target)[0] for b in base_batches]
        achieved = sum(m.byte_mask.sum().item() for m in merged) / max(
            sum(m.boundaries.sum().item() for m in merged), 1
        )
        per_batch = [evaluate(encoder, source, m, source.final_norm) for m in merged]
        agg = {
            k: sum(d[k] for d in per_batch) / len(per_batch)
            for k in per_batch[0]
            if k != "bytes_per_patch"
        }
        row = {"target": target, "achieved": achieved, **agg}
        results.append(row)
        print(
            f"{target:>7.1f} {achieved:>9.2f} {agg['boundary_acc']:>8.4f} "
            f"{agg['sim_corr']:>9.3f} {agg['cos_emb_c']:>10.3f} "
            f"{agg['nn_agree']:>9.3f} {agg['cos_depth']:>7.3f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    print(
        "Note: merged boundaries are applied at evaluation time only. The paper trains\n"
        "with merged supervision (stage 2); these numbers bound the fidelity cost of the\n"
        "patch structure before any training adapts to it."
    )


if __name__ == "__main__":
    main()