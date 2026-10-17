"""Stage 1: subword-to-byte distillation. The source model stays frozen.

    uv run python scripts/train_stage1.py --model jhu-clsp/ettin-encoder-17m \
        --data data/corpus/fineweb-edu.jsonl --steps 20000 --batch-size 8

Trains the local encoder, the boundary predictor and the embedding-alignment head to
recover the source model's behaviour, while the global model is frozen -- the paper's
efficiency trick, which costs roughly a third of end-to-end training.

Because Stage 1 is the paper's proposed vehicle for *architecture* search ("conduct
stage 1 training to rapidly check whether a particular architecture has sufficient
capacity"), this script reports held-out fidelity throughout, not just loss: centred
cosine of pooled embeddings, the correlation between the two models' pairwise
similarity matrices, and nearest-neighbour agreement. Raw cosine is also printed but is
close to uninformative for mean-pooled embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.data import BatchSampler, iter_texts, truncate_to_bytes
from byteify.losses import (
    Stage1Weights,
    boundary_loss,
    embedding_loss,
    stitching_loss,
)
from byteify.source import (
    ByteifyConfig,
    HuggingFaceSourceEncoder,
    LocalEncoderSpec,
    build_byteified_encoder,
)
from byteify.stitching import encode_texts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--data", nargs="*", default=["data/corpus/fineweb-edu.jsonl"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--stitch-depth", type=int, default=-1, help="-1 = full depth")
    p.add_argument("--lambda-b", type=float, default=4.0)
    p.add_argument("--lambda-emb", type=float, default=1.0)
    p.add_argument("--lambda-e", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--empty-cache-every", type=int, default=100)
    p.add_argument("--eval-texts", type=int, default=256)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--local-layers", type=int, default=1)
    p.add_argument("--local-backend", default="attention", choices=["attention", "gru"])
    p.add_argument("--local-window", type=int, default=128)
    p.add_argument("--local-ffn", type=int, default=1024)
    p.add_argument("--boundary-predictor", default="mlp", choices=["mlp", "cosine"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="runs/stage1")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def fidelity(encoder, source, batches) -> dict[str, float]:
    encoder.eval()
    s_embs, t_embs, b_acc, b_tot = [], [], 0.0, 0
    cos_by_depth: dict[int, list[float]] = {}
    for batch in batches:
        out = encoder(
            batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
        )
        depth = source.n_layers
        t_final = source.representation_at(batch.input_ids, batch.attention_mask, depth)
        student = source.forward_layers(out.patch_reps, out.patch_mask, n_layers=depth)

        s = pool_final(student, out.patch_mask, source.final_norm)
        t = pool_final(t_final, batch.attention_mask.bool(), source.final_norm)
        s_embs.append(torch.nn.functional.normalize(s - s.mean(0), dim=-1).cpu())
        t_embs.append(torch.nn.functional.normalize(t - t.mean(0), dim=-1).cpu())

        cos_by_depth.setdefault(depth, []).extend(
            torch.nn.functional.cosine_similarity(student, t_final, dim=-1)[out.patch_mask]
            .flatten()
            .cpu()
            .tolist()
        )

        pred = (torch.sigmoid(out.boundary_logits) > 0.5)[batch.byte_mask.bool()]
        tgt = batch.boundaries[batch.byte_mask.bool()].bool()
        b_acc += (pred == tgt).sum().item()
        b_tot += tgt.numel()

    encoder.train()
    s_all = torch.cat(s_embs)
    t_all = torch.cat(t_embs)
    n = s_all.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    s_sim = s_all @ s_all.T
    t_sim = t_all @ t_all.T
    return {
        "cos_emb_c": (s_all * t_all).sum(-1).mean().item(),
        "sim_corr": pearson(s_sim[iu[0], iu[1]], t_sim[iu[0], iu[1]]).item(),
        "nn_agree": (
            s_sim.fill_diagonal_(float("-inf")).argmax(-1)
            == t_sim.fill_diagonal_(float("-inf")).argmax(-1)
        ).float().mean().item(),
        "nn_chance": 1.0 / max(n - 1, 1),
        "boundary_acc": b_acc / max(b_tot, 1),
        **{f"cos@{d}": sum(v) / len(v) for d, v in cos_by_depth.items()},
        "eval_n": n,
    }


def pool_final(h, mask, final_norm):
    if final_norm is not None:
        h = final_norm(h)
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    cfg = ByteifyConfig(
        model_id=args.model,
        local_layers=args.local_layers,
        local_backend=args.local_backend,
        local_window=args.local_window,
        local_ffn_hidden=args.local_ffn,
        boundary_predictor=args.boundary_predictor,
    )
    print(cfg.describe())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(args.model)).to(device)
    source.eval()
    for p in source.parameters():
        p.requires_grad_(False)

    from byteify.alignment import discover_special_bytes
    from byteify.vocab import resolve_token_bytes

    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    print(
        f"source: {sum(p.numel() for p in source.parameters()) / 1e6:.2f}M params, "
        f"d_model={source.d_model}, layers={source.n_layers}, family={source.family}"
    )
    print(f"tokenizer: {resolution.describe()}, special pseudo-bytes={len(specials)}")

    encoder = build_byteified_encoder(
        source,
        tokenizer,
        LocalEncoderSpec(
            layers=cfg.local_layers,
            backend=cfg.local_backend,
            window=cfg.local_window,
            ffn_hidden=cfg.local_ffn_hidden,
            boundary_predictor=cfg.boundary_predictor,
        ),
    ).to(device)
    encoder.embed.load_suffix_from(source.embedding_matrix())
    # The suffix table is *reused* capacity: it is initialised to the source model's own
    # embedding matrix, which is the whole point of T_SubwordSuffix. Reporting it as new
    # parameters would overstate the local encoder by an order of magnitude.
    n_local_enc = sum(p.numel() for p in encoder.local_encoder.parameters())
    n_boundary = sum(p.numel() for p in (encoder.boundary_predictor.parameters() if encoder.boundary_predictor else []))
    n_suffix = encoder.embed.suffix.weight.numel()
    n_byte_emb = encoder.embed.byte.weight.numel()
    n_source = sum(p.numel() for p in source.parameters())
    print(
        f"byteified: local encoder +{n_local_enc / 1e6:.2f}M, boundary head +{n_boundary / 1e6:.2f}M "
        f"({(n_local_enc + n_boundary) / 1e6 / (n_source / 1e6) * 100:.1f}% of source's {n_source / 1e6:.2f}M); "
        f"byte table +{n_byte_emb / 1e3:.0f}k, suffix table {n_suffix / 1e6:.2f}M reused from source"
    )

    # -- data ---------------------------------------------------------------
    texts = list(iter_texts(args.data, min_chars=64, limit=args.n_train + args.eval_texts))
    if len(texts) < 32:
        raise SystemExit("not enough text; run scripts/prepare_corpus.py first")
    eval_texts, train_texts = texts[: args.eval_texts], texts[args.eval_texts :]
    sampler = BatchSampler(
        train_texts,
        tokenizer,
        batch_size=args.batch_size,
        max_bytes=args.max_bytes,
        add_special_tokens=True,
        seed=args.seed,
    )
    eval_batches = [
        encode_texts(
            [truncate_to_bytes(t, args.max_bytes) for t in eval_texts[i : i + args.batch_size]],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        for i in range(0, min(args.eval_texts, args.eval_batches * args.batch_size), args.batch_size)
    ]
    print(f"data: {len(train_texts)} train / {len(eval_texts)} eval texts\n")

    weights = Stage1Weights(boundary=args.lambda_b, stitching=args.lambda_e, embedding=args.lambda_emb)
    depth = source.n_layers if args.stitch_depth < 0 else args.stitch_depth
    print(f"stitching at depth n={depth}; weights lambda_B={weights.boundary} "
          f"lambda_E={weights.stitching} lambda_emb={weights.embedding}\n")

    params = [p for p in encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "args.json").open("w"), indent=2)

    print(f"{'step':>7} {'total':>9} {'L_B':>8} {'L_E':>9} {'L_emb':>8} {'ms/step':>8}  fidelity")
    history = []
    step = 0
    t_start = time.perf_counter()
    encoder.train()
    while step < args.steps:
        for batch in sampler.epoch():
            if step >= args.steps:
                break
            batch = batch.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            out = encoder(
                batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
            )
            teacher = source.representation_at(batch.input_ids, batch.attention_mask, depth)
            student = source.forward_layers(out.patch_reps, out.patch_mask, n_layers=depth)

            l_b = boundary_loss(out.boundary_logits, batch.boundaries, batch.byte_mask)
            l_e = stitching_loss(student, teacher, out.patch_mask)
            l_emb = embedding_loss(
                student, teacher, out.patch_mask, batch.attention_mask.bool(), source.final_norm
            )
            total = weights.boundary * l_b + weights.stitching * l_e + weights.embedding * l_emb

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            if device == "mps" and args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            if step % args.log_every == 0:
                dt = time.perf_counter() - t_start
                print(
                    f"{step:>7} {total.item():>9.4f} {l_b.item():>8.4f} {l_e.item():>9.4f} "
                    f"{l_emb.item():>8.4f} {dt / max(step, 1) * 1000:>8.0f}"
                )
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                m = fidelity(encoder, source, eval_batches)
                history.append({"step": step, **m})
                print(
                    f"{'':>7} {'':>9} {'':>8} {'':>9} {'':>8} {'':>8}  "
                    f"sim_corr={m['sim_corr']:.3f} cos_emb_c={m['cos_emb_c']:.3f} "
                    f"nn_agree={m['nn_agree']:.3f}/{m['nn_chance']:.3f} "
                    f"bnd={m['boundary_acc']:.3f} cos@{depth}={m[f'cos@{depth}']:.3f}",
                    flush=True,
                )
                json.dump(history, (out_dir / "history.json").open("w"), indent=2)
                torch.save(
                    {"encoder": encoder.state_dict(), "config": vars(args), "step": step},
                    out_dir / "stage1.pt",
                )

    torch.save(
        {"encoder": encoder.state_dict(), "config": vars(args)},
        out_dir / "stage1.pt",
    )
    print(f"\nsaved {out_dir / 'stage1.pt'}; {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()