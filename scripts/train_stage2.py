"""Stage 2: end-to-end byte-level training, with everything unfrozen.

    uv run python scripts/train_stage2.py --model jhu-clsp/ettin-encoder-17m \
        --init runs/stage1_20k/stage1.pt --steps 20000

The paper's Stage 2 drops the distillation losses and keeps only

    L_stage2 = L_B + L_CE

-- the boundary loss and a cross-entropy term -- and unfreezes the global model, whose
job is now to *learn to use byte-level information* rather than to imitate a subword
model. Two deliberate departures for the encoder setting:

* ``L_CE`` is **masked byte prediction**, not next-byte cross-entropy: a bidirectional
  model cannot be trained to predict the next byte, because doing so would leak the
  future into the representation the embeddings are pooled from.
* Boundaries come from the **learned predictor**, not the oracle. Using the oracle would
  leave the model dependent on the subword tokenizer at inference, which is the thing
  byteification is supposed to remove. The cost is that training and inference now agree,
  so boundary error compounds; the boundary loss is what controls it.

Expect fidelity to the source model to *drop* during Stage 2, and that is the point: the
model is trading imitation for byte-level competence. The paper's gains in character
understanding and code come from exactly this trade.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.data import BatchSampler, iter_texts, truncate_to_bytes
from byteify.losses import boundary_loss, masked_byte_loss, sample_mlm_mask
from byteify.model import ByteMLMHead
from byteify.source import (
    HuggingFaceSourceEncoder,
    LocalEncoderSpec,
    build_byteified_encoder,
)
from byteify.stitching import encode_texts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--init", default=None, help="Stage 1 checkpoint to continue from")
    p.add_argument("--data", nargs="*", default=["data/corpus/fineweb-edu.jsonl"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--local-lr-mult", type=float, default=10.0, help="LR multiplier for new params")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lambda-b", type=float, default=4.0)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--out-layers", type=int, default=2)
    p.add_argument("--out-ffn", type=int, default=1024)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--empty-cache-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="runs/stage2")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def evaluate(encoder, head, source, batches, args) -> dict[str, float]:
    encoder.eval()
    head.eval()
    ce_sum, ce_n = 0.0, 0
    b_ok, b_n = 0, 0
    for batch in batches:
        inputs, mask = sample_mlm_mask(batch.byte_ids, batch.byte_mask, mlm_prob=args.mlm_prob)
        out = encoder(inputs, batch.expanded_ids, batch.byte_mask)
        logits = head(out.byte_context)
        loss = masked_byte_loss(logits, batch.byte_ids, mask)
        k = int(mask.sum().item())
        ce_sum += loss.item() * max(k, 1)
        ce_n += k
        pred = (torch.sigmoid(out.boundary_logits) > 0.5)[batch.byte_mask.bool()]
        tgt = batch.boundaries[batch.byte_mask.bool()].bool()
        b_ok += (pred == tgt).sum().item()
        b_n += tgt.numel()
    encoder.train()
    head.train()
    ce = ce_sum / max(ce_n, 1)
    return {
        "mlm_ce": ce,
        "byte_ppl": math.exp(min(ce, 20)),
        "boundary_acc": b_ok / max(b_n, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(args.model)).to(device)

    # The interface's architecture comes out of the checkpoint when there is one, and out
    # of the declared flags otherwise -- never out of constants written at this call site,
    # which silently built a different shape for any checkpoint trained with other settings.
    if args.init:
        recorded = torch.load(args.init, map_location="cpu", weights_only=False)
        spec = LocalEncoderSpec.from_config(recorded.get("config") or recorded.get("args") or {})
        print(f"loaded Stage 1 interface from {args.init}: {spec}")
    else:
        spec = LocalEncoderSpec(boundary_predictor="mlp")
        print(f"no --init given: building the byte interface fresh: {spec}")

    # Stage 2 predicts masked bytes off the *output* side, so the local decoder has to
    # exist. A Stage 1 checkpoint has none -- that model was trained without it -- so the
    # flag decides the output side rather than the checkpoint. The rest of the interface
    # still comes from the checkpoint, so Stage 1's learned weights are kept.
    spec = replace(spec, out_layers=args.out_layers, out_ffn_hidden=args.out_ffn)

    encoder = build_byteified_encoder(source, tokenizer, spec).to(device)
    if args.init:
        _, unexpected = encoder.load_state_dict(recorded["encoder"], strict=False)
        if unexpected:
            raise SystemExit(f"checkpoint holds weights this encoder cannot use: {sorted(unexpected)}")
        encoder.embed.load_suffix_from(source.embedding_matrix())
    else:
        encoder.embed.load_suffix_from(source.embedding_matrix())
        print("initialised the suffix table from the source embeddings")

    head = ByteMLMHead(source.d_model).to(device)
    print(f"byte interface: {spec.layers}x{spec.backend} ffn={spec.ffn_hidden} "
          f"win={spec.window}, out_layers={spec.out_layers}")

    for p in source.parameters():
        p.requires_grad_(True)
    source.train()

    texts = list(iter_texts(args.data, min_chars=64, limit=args.n_train))
    sampler = BatchSampler(
        texts, tokenizer, batch_size=args.batch_size, max_bytes=args.max_bytes, seed=args.seed
    )
    eval_batches = [
        encode_texts(
            [truncate_to_bytes(t, args.max_bytes) for t in texts[i * args.batch_size : (i + 1) * args.batch_size]],
            tokenizer,
            device=device,
        )
        for i in range(args.eval_batches)
    ]
    print(f"data: {len(texts)} texts; out_layers={args.out_layers}; device={device}\n")

    new_params = list(encoder.parameters()) + list(head.parameters())
    new_ids = {id(p) for p in new_params}
    old_params = [p for p in source.parameters() if id(p) not in new_ids]
    opt = torch.optim.AdamW(
        [
            {"params": old_params, "lr": args.lr},
            {"params": new_params, "lr": args.lr * args.local_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    # `new_params` are freshly added; the paper trains them harder relative to the
    # pretrained global model, which is otherwise close to its pretraining optimum.

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "args.json").open("w"), indent=2)

    print(f"{'step':>7} {'total':>9} {'L_B':>8} {'L_CE':>9} {'bpc':>7} {'ms/step':>8}  eval")
    history = []
    step = 0
    t0 = time.perf_counter()
    while step < args.steps:
        for batch in sampler.epoch():
            if step >= args.steps:
                break
            batch = batch.to(device)
            scale = lr_scale(step)
            opt.param_groups[0]["lr"] = args.lr * scale
            opt.param_groups[1]["lr"] = args.lr * args.local_lr_mult * scale

            inputs, mask = sample_mlm_mask(batch.byte_ids, batch.byte_mask, mlm_prob=args.mlm_prob)
            # No oracle boundaries: the learned predictor decides the patches, so the
            # model is deployable without the subword tokenizer.
            out = encoder(inputs, batch.expanded_ids, batch.byte_mask)
            logits = head(out.byte_context)

            l_b = boundary_loss(out.boundary_logits, batch.boundaries, batch.byte_mask)
            l_ce = masked_byte_loss(logits, batch.byte_ids, mask)
            total = args.lambda_b * l_b + args.lambda_ce * l_ce

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()) + old_params, args.grad_clip
            )
            opt.step()

            if device == "mps" and args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            if step % args.log_every == 0:
                dt = time.perf_counter() - t0
                print(
                    f"{step:>7} {total.item():>9.4f} {l_b.item():>8.4f} {l_ce.item():>9.4f} "
                    f"{l_ce.item() / max(math.log(256), 1e-9):>7.3f} {dt / max(step, 1) * 1000:>8.0f}",
                    flush=True,
                )
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(encoder, head, source, eval_batches, args)
                history.append({"step": step, **m})
                print(
                    f"{'':>7} {'':>9} {'':>8} {'':>9} {'':>7} {'':>8}  "
                    f"byte_ppl={m['byte_ppl']:.3f} boundary_acc={m['boundary_acc']:.4f}",
                    flush=True,
                )
                json.dump(history, (out_dir / "history.json").open("w"), indent=2)
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        # Stage 2 unfreezes and trains the global model, so a checkpoint
                        # that omits it is not resumable and not loadable -- it would
                        # silently serve a global model that was never trained.
                        "source": source.state_dict(),
                        "args": vars(args),
                        "step": step,
                    },
                    out_dir / "stage2.pt",
                )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "source": source.state_dict(),
            "args": vars(args),
        },
        out_dir / "stage2.pt",
    )
    print(f"\nsaved {out_dir / 'stage2.pt'}; {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()