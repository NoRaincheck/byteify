"""Fidelity probe: can a byte-level local encoder stand in for the subword model?

This is Stage 1's diagnostic, and the cheapest possible test of the whole byteification
bet. Bolmo's paper describes stage 1 as a way to "rapidly check whether a particular
architecture for the local encoder and decoder has sufficient capacity to emulate the
input and output embedding matrices" -- and to decide architecture questions from that,
rather than after full training.

For an encoder there is no output embedding matrix to emulate, so the question becomes:
can the local encoder's pooled patch representations reproduce the *representations*
of the frozen source encoder?

We report, for several stitching depths ``n``:

* ``L_E(n)``  masked MSE between the student's and teacher's representations at layer n.
* ``cos(n)``   mean cosine similarity at layer n.
* ``cos_emb``  cosine similarity between the two models' *sentence* embeddings, after
  mean pooling over patches/tokens -- the quantity an embedding model is actually judged on.

Two configurations are reported: the local encoder at random init, and after a short
stage-1 run. If training does not move ``cos_emb`` towards 1, the architecture is
undersized and no amount of later training will rescue it.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from byteify.alignment import discover_special_bytes
from byteify.model import ByteifiedEncoder, LocalContextEncoder, PartialModernBert
from byteify.source import resolve_source_encoder
from byteify.stitching import encode_texts, masked_l2, teacher_states
from byteify.vocab import resolve_token_bytes

DEPTHS = (0, 1, 2, 4, 7)


def load_corpus(paths: list[Path], n_texts: int, max_bytes: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts: list[str] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Split on blank lines so each unit is a coherent paragraph.
        for para in raw.split("\n\n"):
            para = " ".join(para.split())
            if len(para) < 80:
                continue
            texts.append(para[:max_bytes])
    rng.shuffle(texts)
    return texts[:n_texts]


def sentence_embedding(hidden: torch.Tensor, mask: torch.Tensor, final_norm=None) -> torch.Tensor:
    """Mean pooling over valid positions, matching the model's mean pooling.

    ``final_norm`` is ModernBERT's output RMSNorm, which the real embedding path
    applies *before* pooling. Stitching on pre-norm states is not the same target as
    the representation an embedding model is judged on, so we normalise here.
    """
    if final_norm is not None:
        hidden = final_norm(hidden)
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)


@torch.no_grad()
def evaluate(
    encoder: ByteifiedEncoder,
    partial: PartialModernBert,
    batches: list,
    device: str,
) -> dict[str, float]:
    out_all: list = []
    teacher_all: list = []
    metrics: dict[str, float] = {}
    bytes_n = patches_n = 0

    for batch in batches:
        out = encoder(
            batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
        )
        teacher = teacher_states(
            partial.model, batch.input_ids, batch.attention_mask, n_layers=DEPTHS
        )
        final_student = partial(out.patch_reps, out.patch_mask, n_layers=DEPTHS[-1])
        final_teacher = teacher[DEPTHS[-1]]

        s_emb = sentence_embedding(final_student, out.patch_mask, partial.model.final_norm)
        t_emb = sentence_embedding(
            final_teacher, batch.attention_mask.bool(), partial.model.final_norm
        )
        out_all.append(F.normalize(s_emb, dim=-1).cpu())
        teacher_all.append(F.normalize(t_emb, dim=-1).cpu())
        bytes_n += batch.byte_mask.sum().item()
        patches_n += batch.boundaries.sum().item()

        for n in DEPTHS:
            s = partial(out.patch_reps, out.patch_mask, n_layers=n)
            key = f"{n}"
            metrics[f"mse@{key}"] = metrics.get(f"mse@{key}", 0.0) + masked_l2(
                s, teacher[n], out.patch_mask
            ).item()
            metrics[f"cos@{key}"] = metrics.get(f"cos@{key}", 0.0) + F.cosine_similarity(
                s, teacher[n], dim=-1
            )[out.patch_mask].mean().item()

    nb = len(batches)
    for k in list(metrics):
        metrics[k] /= nb

    s_all = torch.cat(out_all)
    t_all = torch.cat(teacher_all)
    metrics["cos_emb"] = (s_all * t_all).sum(-1).mean().item()

    # Mean-pooled embeddings are strongly anisotropic: they share one dominant
    # direction, so raw cosine sits near 1.0 even for unrelated texts. Centering
    # removes that common component and leaves the part that actually discriminates.
    # Reporting raw cosine alone is misleading -- it can read 0.99 while the two
    # models agree on nearest neighbours no better than chance.
    sc = s_all - s_all.mean(0, keepdim=True)
    tc = t_all - t_all.mean(0, keepdim=True)
    metrics["cos_emb_c"] = (
        F.cosine_similarity(sc, tc, dim=-1).mean().item()
    )

    n = s_all.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    s_c = sc / sc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t_c = tc / tc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    s_sim = s_c @ s_c.T
    t_sim = t_c @ t_c.T
    metrics["sim_corr"] = pearson(
        s_sim[iu[0], iu[1]], t_sim[iu[0], iu[1]]
    ).item()

    s_nn = s_sim.fill_diagonal_(float("-inf")).argmax(-1)
    t_nn = t_sim.fill_diagonal_(float("-inf")).argmax(-1)
    metrics["nn_agree"] = (s_nn == t_nn).float().mean().item()
    metrics["nn_chance"] = 1.0 / max(n - 1, 1)

    metrics["bytes_per_patch"] = bytes_n / max(patches_n, 1)
    metrics["eval_texts"] = n
    return metrics


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)


def show(tag: str, m: dict[str, float]) -> None:
    tracked = [
        "cos_emb",
        "cos_emb_c",
        "sim_corr",
        "nn_agree",
        "cos@7",
        "mse@7",
        "bytes_per_patch",
    ]
    parts = [f"{k}={m[k]:.4f}" for k in tracked if k in m]
    print(f"  {tag:<20} {'  '.join(parts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    parser.add_argument("--data", nargs="*", default=sorted(Path("data").glob("pg*.txt")))
    parser.add_argument("--n-texts", type=int, default=32)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--stitch-depth", type=int, default=4)
    parser.add_argument("--lambda-b", type=float, default=4.0, help="boundary loss weight")
    parser.add_argument(
        "--lambda-emb",
        type=float,
        default=0.0,
        help="weight on direct sentence-embedding cosine alignment",
    )
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--backend", default="attention", choices=["attention", "gru"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = AutoModel.from_pretrained(args.model).to(device).eval()
    for p in source.parameters():
        p.requires_grad_(False)

    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    partial = PartialModernBert(source)
    d_model = source.config.hidden_size

    encoder = ByteifiedEncoder(
        d_model=d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        local_encoder=LocalContextEncoder(
            d_model,
            n_layers=args.local_layers,
            ffn_hidden=args.ffn_hidden,
            n_heads=source.config.num_attention_heads,
            window=args.window,
            backend=args.backend,
        ),
        boundary_predictor="mlp",
    ).to(device)
    encoder.embed.load_suffix_from(resolve_source_encoder(source).embedding_matrix().data)

    texts = load_corpus([Path(p) for p in args.data], args.n_texts, args.max_bytes, args.seed)

    # Hold out the tail for evaluation. Without this the fidelity numbers just
    # measure memorization: 16 texts is a handful of kilobytes and the local
    # encoder is being fit directly against them.
    n_eval = max(1, len(texts) // 4)
    eval_texts, train_texts = texts[:n_eval], texts[n_eval:]
    batches = [
        encode_texts(train_texts[i : i + args.batch_size], tokenizer, device=device)
        for i in range(0, len(train_texts), args.batch_size)
    ]
    eval_batches = [
        encode_texts(eval_texts[i : i + args.batch_size], tokenizer, device=device)
        for i in range(0, len(eval_texts), args.batch_size)
    ]

    n_local = sum(p.numel() for p in encoder.local_encoder.parameters())
    print(f"source {args.model}: {sum(p.numel() for p in source.parameters()) / 1e6:.1f}M params")
    print(f"local encoder: {n_local / 1e6:.2f}M params, backend={args.backend}, window={args.window}")
    print(f"{len(texts)} texts ({len(train_texts)} train / {len(eval_texts)} held out), "
          f"stitching depth n={args.stitch_depth}\n")

    print("stage 1 diagnostics")
    show("random init", evaluate(encoder, partial, eval_batches, device))

    if args.eval_only:
        return

    # Stage 1: global model frozen, only the local encoder (and boundary head) train.
    for p in source.parameters():
        p.requires_grad_(False)
    params = [p for p in encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1)

    print(f"\ntraining local encoder for {args.steps} steps")
    t0 = time.perf_counter()
    step = 0
    while step < args.steps:
        for batch in batches:
            if step >= args.steps:
                break
            out = encoder(
                batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
            )
            teacher = teacher_states(
                partial.model, batch.input_ids, batch.attention_mask, n_layers=[args.stitch_depth]
            )[args.stitch_depth]
            student = partial(out.patch_reps, out.patch_mask, n_layers=args.stitch_depth)
            loss_e = masked_l2(student, teacher, out.patch_mask)

            # The encoder analog of Bolmo's L_{D,Distill}: align the model's actual
            # *output behaviour*. For an LM that is next-token likelihood; for an
            # embedder it is the pooled sentence embedding, so we align that cosine
            # directly. L2 on hidden states does not target it -- scale errors that
            # L2 punishes are invisible to cosine, and vice versa.
            loss_emb = torch.zeros((), device=device)
            if args.lambda_emb > 0:
                s_emb = sentence_embedding(
                    student, out.patch_mask, partial.model.final_norm
                )
                t_emb = sentence_embedding(
                    teacher, batch.attention_mask.bool(), partial.model.final_norm
                )
                # Batch-centre both sides before taking cosine. Pooled embeddings
                # share a dominant direction that raw cosine rewards and that carries
                # almost no information; centring makes the loss respond to the
                # component that actually separates documents.
                s_c = s_emb - s_emb.mean(0, keepdim=True)
                t_c = t_emb - t_emb.mean(0, keepdim=True)
                loss_emb = 1.0 - F.cosine_similarity(s_c, t_c, dim=-1).mean()

            # Stage 1 trains the boundary predictor and the local encoder together,
            # with lambda_B = 4 as in the paper.
            valid = batch.byte_mask.bool()
            loss_b = F.binary_cross_entropy_with_logits(
                out.boundary_logits[valid], batch.boundaries[valid].float()
            )
            loss = args.lambda_b * loss_b + loss_e + args.lambda_emb * loss_emb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % 50 == 0:
                print(
                    f"  step {step:4d}  L={loss.item():.5f}"
                    f"  L_E={loss_e.item():.5f}  L_B={loss_b.item():.5f}"
                    f"  L_emb={loss_emb.item():.5f}"
                )
            step += 1
    dt = time.perf_counter() - t0
    print(f"  {args.steps} steps in {dt:.1f}s ({dt / max(step, 1) * 1000:.0f} ms/step on {device})")

    print()
    show("after stage 1", evaluate(encoder, partial, eval_batches, device))

    # The boundary predictor's own Stage 1 loss: BCE against B_subword.
    enc = encoder.eval()
    with torch.no_grad():
        out = enc(batch.byte_ids, batch.expanded_ids, batch.byte_mask)
        bce = F.binary_cross_entropy_with_logits(
            out.boundary_logits[batch.byte_mask.bool()],
            batch.boundaries[batch.byte_mask.bool()].float(),
        )
        pred = (torch.sigmoid(out.boundary_logits) > 0.5)
        target = batch.boundaries.bool()
        acc = (pred[target] == target[target]).float().mean().item()
        pos_recall = (pred & target)[target].float().mean().item()
    print(f"\nboundary predictor: BCE={bce.item():.4f} accuracy={acc:.4f} positive-recall={pos_recall:.4f}")


if __name__ == "__main__":
    main()