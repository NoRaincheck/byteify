"""Measure a source model's fidelity floors, so ``byteify build`` can be pointed at it.

Thresholds are published as data because fidelity numbers do not transfer between
architectures (ADR 0010). That makes adding a source model a measurement rather than a
decision, and this is the measurement: it runs the *gate's own* code path -- budgeted
patching, the artifact's byte axis and patch budget -- because a number measured any other
way is a number about a model nobody will deploy.

It reports four things, and the last one is the reason a floor can be chosen at all:

1. a randomly initialised byte interface, which exports cleanly and passes every check
   except fidelity, and is therefore the contrast every real number has to beat;
2. a Stage 1 checkpoint, if one is given;
3. suffix-trie coverage over the same texts;
4. the *source model's own* retrieval on the judgements, because a retrieval floor is
   only interpretable next to what the model being reproduced manages.

Floors are then chosen below the observed values and far above the random-init ones, and
written into ``src/byteify/assets/thresholds.json`` by hand -- a measurement script that
edited its own acceptance criteria would not be a measurement.

    uv run python scripts/measure_thresholds.py \
        --model hotchpotch/bekko-system-one-v0-17m \
        --data data/corpus/open-jev-train.jsonl \
        --checkpoint runs/stage1_bekko/stage1.pt \
        --judgements data/judgements/open-jev-validation.jsonl \
        --n-eval 64 --budget 128 --max-bytes 512
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from byteify.alignment import discover_special_bytes
from byteify.contrastive import retrieval_over_judgements
from byteify.data import truncate_to_bytes
from byteify.fidelity import _budgeted_patch_reps, measure_fidelity, pool_final
from byteify.gates import assets_path
from byteify.hub import resolve_source_model, resolve_source_tokenizer
from byteify.inference import ByteifiedEmbedder
from byteify.manifest import measure_trie_coverage
from byteify.source import (
    HuggingFaceSourceEncoder,
    LocalEncoderSpec,
    build_byteified_encoder,
    load_encoder_state,
)
from byteify.stitching import encode_texts
from byteify.training import load_judgements, load_reference_texts
from byteify.vocab import resolve_token_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="a Stage 1 checkpoint; without one, only the random-init "
                             "baseline is reported")
    parser.add_argument("--judgements", default=None)
    parser.add_argument("--n-eval", type=int, default=64,
                        help="held-out texts to grade, taken from the head of --data: "
                             "that is the slice the gate reads and the slice Stage 1 "
                             "holds out, so this measures what the gate will measure")
    parser.add_argument("--min-texts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--local-ffn", type=int, default=1024)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write the measurements as JSON")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = resolve_source_tokenizer(args.model)
    resolved = resolve_source_model(args.model)
    source = HuggingFaceSourceEncoder(
        resolved.model, model_id=args.model, layout=resolved.layout
    ).to(device).eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    token_bytes = {i: b for i, b in enumerate(resolution.token_bytes) if b}

    texts = [truncate_to_bytes(t, args.max_bytes) for t in load_reference_texts(args.data)]
    if len(texts) < max(args.n_eval, args.min_texts):
        raise SystemExit(
            f"--data yielded {len(texts)} usable texts; at least {max(args.n_eval, args.min_texts)} "
            "are needed to hold some out"
        )
    eval_texts = texts[: args.n_eval]
    coverage = measure_trie_coverage(token_bytes, eval_texts)

    encoder = build_byteified_encoder(
        source,
        tokenizer,
        LocalEncoderSpec(
            layers=args.local_layers,
            ffn_hidden=args.local_ffn,
            window=args.local_window,
            boundary_predictor="mlp",
        ),
    ).to(device)
    encoder.embed.load_suffix_from(source.embedding_matrix())

    batches = [
        encode_texts(eval_texts[i : i + args.batch_size], tokenizer, device=device)
        for i in range(0, len(eval_texts) - args.batch_size + 1, args.batch_size)
    ]

    print(f"source     {args.model}")
    print(f"  layout   {resolved.layout}"
          + (f" ({resolved.module_path})" if resolved.module_path else ""))
    print(f"  size     {sum(p.numel() for p in source.parameters()) / 1e6:.2f}M params, "
          f"{source.family}, d_model={source.d_model}, layers={source.n_layers}")
    print(f"  tokenizer {resolution.describe()}, {len(specials)} pseudo-byte special(s)")
    print(f"evaluation {len(eval_texts)} held-out texts, {len(batches)} batches, "
          f"byte axis {args.max_bytes}, patch budget {args.budget}")
    print(f"trie coverage over the evaluation texts: {coverage:.4f}")

    measurements: dict[str, object] = {
        "source_id": args.model,
        "architecture": resolved.layout,
        "eval_texts": len(eval_texts),
        "budget": args.budget,
        "max_bytes": args.max_bytes,
        "trie_coverage": coverage,
        "vocab_strategy": resolution.strategy,
    }

    def measure(tag: str) -> dict[str, float]:
        embedder = ByteifiedEmbedder(
            encoder, source, n_patches=min(args.budget, args.max_bytes)
        ).to(device).eval()
        # Floors of zero, so nothing here can pass or fail: this is a ruler, not a gate.
        metrics = measure_fidelity(
            lambda batch: _budgeted_patch_reps(embedder, batch), source, batches
        )
        print(f"\n{tag}")
        print(f"  cos_emb_c   {metrics['cos_emb_c']:+.4f}   (raw {metrics['cos_emb_raw']:.4f})")
        print(f"  sim_corr    {metrics['sim_corr']:+.4f}")
        print(f"  boundary_acc {metrics['boundary_acc']:.4f}")
        print(f"  nn_agree    {metrics['nn_agree']:.4f} against chance {metrics['nn_chance']:.4f}")
        return metrics

    measurements["random_init"] = measure("random init (the contrast every floor must beat)")

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["encoder"]
        load_encoder_state(encoder, state, source=str(args.checkpoint))
        encoder.eval()
        measurements["trained"] = measure(f"trained ({args.checkpoint})")

    if args.judgements:
        judgements = load_judgements(args.judgements)
        embedder = ByteifiedEmbedder(
            encoder, source, n_patches=min(args.budget, args.max_bytes)
        ).to(device).eval()

        def source_embed(texts: list[str]) -> torch.Tensor:
            rows = encode_texts(
                [truncate_to_bytes(t, args.max_bytes) for t in texts],
                tokenizer, device=device,
            )
            with torch.no_grad():
                return pool_final(
                    source.representation_at(rows.input_ids, rows.attention_mask, source.n_layers),
                    rows.attention_mask.bool(),
                    source.final_norm,
                )

        source_retrieval = retrieval_over_judgements(
            source_embed, judgements, batch_size=args.batch_size, max_bytes=args.max_bytes
        )
        print(f"\nsource model's own retrieval on {args.judgements}")
        print(f"  recall@1   {source_retrieval['recall@1']:.4f}   "
              f"(chance {source_retrieval['chance_recall_at_1']:.4f})")
        print(f"  mrr        {source_retrieval['mrr']:.4f}")
        print(f"  ndcg       {source_retrieval['ndcg']:.4f}")
        print(f"  decisions  {source_retrieval['n_decisions']}")
        measurements["source_retrieval"] = source_retrieval

        trained = "trained" in measurements
        artifact_retrieval = retrieval_over_judgements(
            lambda texts: embedder.encode(tokenizer, texts, max_bytes=args.max_bytes),
            judgements, batch_size=args.batch_size, max_bytes=args.max_bytes,
        )
        tag = "byteified artifact" if trained else "random init"
        print(f"\n{tag} retrieval")
        print(f"  recall@1   {artifact_retrieval['recall@1']:.4f}   "
              f"(chance {artifact_retrieval['chance_recall_at_1']:.4f})")
        print(f"  mrr        {artifact_retrieval['mrr']:.4f}")
        measurements[f"{'trained' if trained else 'random_init'}_retrieval"] = (
            artifact_retrieval
        )

    print("\nFloors go below the trained numbers above and far above the random-init ones.")
    print(f"Edit {assets_path('thresholds.json')} and record both.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(measurements, handle, indent=2, sort_keys=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    random.seed(0)
    main()