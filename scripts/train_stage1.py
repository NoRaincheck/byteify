"""Stage 1: subword-to-byte distillation. The source model stays frozen.

    uv run python scripts/train_stage1.py --model jhu-clsp/ettin-encoder-17m \
        --data data/corpus/fineweb-edu.jsonl --steps 20000 --batch-size 8

Trains the local encoder, the boundary predictor and the embedding-alignment head to
recover the source model's behaviour, while the global model is frozen -- the paper's
efficiency trick, which costs roughly a third of end-to-end training.

This is a research script. The product path is ``byteify build``, which runs the same
training through :func:`byteify.training.train_stage1` and then gates the result. What
this script adds is a run history on disk and a checkpoint at a path you choose, which is
what you want when the run is an experiment rather than a deliverable.

The recipe itself lives in the package, not here, so the two cannot drift into training
two different things.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.data import iter_texts
from byteify.source import HuggingFaceSourceEncoder, LocalEncoderSpec, build_byteified_encoder
from byteify.training import Stage1Config, save_checkpoint, train_stage1


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(
        AutoModel.from_pretrained(args.model), model_id=args.model
    ).to(device)
    source.eval()
    for p in source.parameters():
        p.requires_grad_(False)

    spec = LocalEncoderSpec(
        layers=args.local_layers,
        backend=args.local_backend,
        window=args.local_window,
        ffn_hidden=args.local_ffn,
        boundary_predictor=args.boundary_predictor,
    )
    print(f"source: {sum(p.numel() for p in source.parameters()) / 1e6:.2f}M params, "
          f"d_model={source.d_model}, layers={source.n_layers}, family={source.family}")
    print(f"byte interface: {spec.layers}x{spec.backend} ffn={spec.ffn_hidden} "
          f"window={spec.window}, boundary={spec.boundary_predictor}")

    encoder = build_byteified_encoder(source, tokenizer, spec).to(device)
    encoder.embed.load_suffix_from(source.embedding_matrix())
    # The suffix table is *reused* capacity: it is initialised to the source model's own
    # embedding matrix, which is the whole point of T_SubwordSuffix. Reporting it as new
    # parameters would overstate the local encoder by an order of magnitude.
    n_local = sum(p.numel() for p in encoder.local_encoder.parameters())
    n_boundary = sum(p.numel() for p in encoder.boundary_predictor.parameters())
    n_source = sum(p.numel() for p in source.parameters())
    print(f"byteified: local encoder +{n_local / 1e6:.2f}M, boundary head "
          f"+{n_boundary / 1e6:.2f}M ({n_local / n_source * 100:.1f}% of the source's "
          f"{n_source / 1e6:.2f}M)")

    texts = list(iter_texts(args.data, min_chars=64, limit=args.n_train + args.eval_texts))
    if len(texts) < 64:
        raise SystemExit(
            f"reference data yielded {len(texts)} usable texts; at least 64 are needed "
            "to hold some out. Run scripts/prepare_corpus.py, or pass more --data."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "args.json").open("w"), indent=2)

    cfg = Stage1Config(
        steps=args.steps,
        batch_size=args.batch_size,
        max_bytes=args.max_bytes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        stitch_depth=args.stitch_depth,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_texts=args.eval_texts,
        n_train=args.n_train,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=device,
        log_every=args.log_every,
    )
    from byteify.losses import Stage1Weights

    cfg.weights = Stage1Weights(
        boundary=args.lambda_b, stitching=args.lambda_e, embedding=args.lambda_emb
    )

    started = time.perf_counter()
    result = train_stage1(
        encoder, source, tokenizer, texts, cfg, log=lambda m: print(m, flush=True)
    )

    json.dump(result.history, (out_dir / "history.json").open("w"), indent=2)
    # `spec.to_config()` is what makes the checkpoint loadable: it records the interface's
    # own architecture under the names the trainers write. A manifest built from this
    # checkpoint reads exactly these keys.
    save_checkpoint(
        out_dir / "stage1.pt",
        result.state_dict,
        {**result.config, "model": args.model, **spec.to_config(), "seed": args.seed},
    )
    print(f"\nsaved {out_dir / 'stage1.pt'}; {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()