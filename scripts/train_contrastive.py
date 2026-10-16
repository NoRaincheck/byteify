"""Contrastive fine-tuning: the stage that makes the output useful as an embedding.

    uv run python scripts/train_contrastive.py --init runs/stage1/stage1.pt --steps 500

Stage 1 recovers the source model's behaviour and Stage 2 teaches byte-level competence,
but neither optimises retrieval: both target *representations*, not the similarity
structure an embedding model is judged on. This script supplies the third stage --
symmetric InfoNCE with in-batch negatives, trained on the budgeted patch axis used at
inference so there is no train/deploy mismatch.

Positives are mined locally with the source model (:func:`byteify.contrastive.mine_pairs`),
so no external dataset is required.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.contrastive import ContrastiveHead, infonce, mine_pairs, pair_batches
from byteify.data import iter_texts
from byteify.inference import (
    ByteifiedEmbedder,
)
from byteify.model import ByteifiedEncoder, LocalContextEncoder
from byteify.source import HuggingFaceSourceEncoder
from byteify.stitching import encode_texts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--init", default=None, help="Stage 1/2 checkpoint")
    p.add_argument("--data", nargs="*", default=["data/corpus/fineweb-edu.jsonl"])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-bytes", type=int, default=256)
    p.add_argument("--n-patches", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--local-lr-mult", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj-dim", type=int, default=0, help="0 = no projection head")
    p.add_argument("--pairs-per-anchor", type=int, default=1)
    p.add_argument("--pool-size", type=int, default=64)
    p.add_argument("--mine-texts", type=int, default=2000)
    p.add_argument("--freeze-source", action="store_true", default=True)
    p.add_argument("--unfreeze-source", dest="freeze_source", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="runs/contrastive")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--empty-cache-every", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def source_embeddings(texts, tokenizer, source, batch_size=16, max_bytes=1024, device="cpu"):
    """Embed text with the *source* model, for pair mining."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_bytes, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = source.forward_layers(
            source._embeddings(input_ids=enc["input_ids"]),
            enc["attention_mask"],
            n_layers=source.n_layers,
        )
        if source.final_norm is not None:
            hidden = source.final_norm(hidden)
        m = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
        out.append(pooled.cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(args.model)).to(device).eval()

    from byteify.alignment import discover_special_bytes
    from byteify.vocab import resolve_token_bytes

    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)

    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=1024, n_heads=source.n_attention_heads, window=128
        ),
        boundary_predictor="mlp",
    ).to(device)

    if args.init and Path(args.init).exists():
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        print(f"initialised from {args.init}")
    else:
        encoder.embed.load_suffix_from(source.embedding_matrix())
        print("no --init given; suffix table initialised from the source embeddings")

    head = ContrastiveHead(source.d_model, proj_dim=args.proj_dim).to(device)
    embedder = ByteifiedEmbedder(encoder, source, n_patches=args.n_patches)

    if args.freeze_source:
        for p in source.parameters():
            p.requires_grad_(False)

    texts = list(iter_texts(args.data, min_chars=64, limit=args.mine_texts))
    print(f"mining positives from {len(texts)} texts with the source model...")
    pairs = mine_pairs(
        texts,
        lambda batch: source_embeddings(
            batch, tokenizer, source, args.batch_size, args.max_bytes * 2, device
        ),
        pool_size=args.pool_size,
        n_per_anchor=args.pairs_per_anchor,
        seed=args.seed,
    )
    print(f"mined {len(pairs)} pairs")

    new_params = list(encoder.parameters()) + list(head.parameters())
    new_ids = {id(p) for p in new_params}
    old_params = [p for p in source.parameters() if id(p) not in new_ids and p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": new_params, "lr": args.lr * args.local_lr_mult},
            {"params": old_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "args.json").open("w"), indent=2)

    print(f"\n{'step':>6} {'loss':>9} {'temp':>7} {'acc':>6} {'ms/step':>8}")
    step = 0
    t0 = time.perf_counter()
    history = []

    while step < args.steps:
        for batch_pairs in pair_batches(pairs, args.batch_size, seed=args.seed):
            if step >= args.steps:
                break
            # Truncate before encoding: the local encoder builds a dense B*N^2 mask,
            # so the byte length per row is the real memory constraint, and mined
            # positives are full documents rather than the trimmed training segments.
            clip = lambda ts: [t.encode("utf-8")[: args.max_bytes].decode("utf-8", errors="ignore") for t in ts]
            ba = encode_texts(clip([a for a, _ in batch_pairs]), tokenizer, device=device)
            bp = encode_texts(clip([b for _, b in batch_pairs]), tokenizer, device=device)

            scale = lr_scale(step)
            opt.param_groups[0]["lr"] = args.lr * args.local_lr_mult * scale
            opt.param_groups[1]["lr"] = args.lr * scale

            ea = embedder(ba.byte_ids, ba.expanded_ids, ba.byte_mask)
            ep = embedder(bp.byte_ids, bp.expanded_ids, bp.byte_mask)
            pa, temp = head(ea)
            pp, _ = head(ep)

            loss = infonce(pa, pp, temperature=temp)

            opt.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(new_params + old_params, 1.0)
            opt.step()

            if step % args.log_every == 0:
                dt = time.perf_counter() - t0
                print(
                    f"{step:>6} {loss.total.item():>9.4f} {float(temp.detach()):>7.4f} "
                    f"{loss.accuracy.item():>6.3f} {dt / max(step, 1) * 1000:>8.0f}",
                    flush=True,
                )
            history.append(
                {"step": step, "loss": loss.total.item(), "temperature": float(temp.detach()),
                 "accuracy": loss.accuracy.item()}
            )
            step += 1
            if device == "mps" and args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.mps.empty_cache()

    torch.save(
        {"encoder": encoder.state_dict(), "head": head.state_dict(), "args": vars(args)},
        out_dir / "contrastive.pt",
    )
    json.dump(history, (out_dir / "history.json").open("w"), indent=2)
    print(f"\nsaved {out_dir / 'contrastive.pt'}; {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()