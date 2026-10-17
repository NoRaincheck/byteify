"""Where does level 2 stop being level 1?

Level 2 declares that it guarantees everything `faithful` guarantees and, on top of that,
retrieval quality on the supplied judgements. Those two claims were measured against each
other on `bekko-system-one-v0-17m` and they conflict: contrastive fitting lifts recall@1 on
the decision corpus from 0.30 to 0.55, and spends centred cosine 0.90 -> 0.60 to do it.
Under the recipe in :func:`byteify.contrastive.train_contrastive_on_corpus` the level-2
build therefore fails three of its own gates and writes nothing, which is the gate working
as designed and the ladder as declared being unclimbable.

This measures the trade-off curve rather than asserting either end of it. It starts from
one Stage 1 checkpoint, applies contrastive training at a ladder of step counts and
learning rates, and reports both sides of the trade after each point: the level-1 metrics
that must not fall below their floors, and recall@1, which must reach its floor.

    uv run python scripts/contrastive_tradeoff.py \
        --model hotchpotch/bekko-system-one-v0-17m \
        --checkpoint runs/stage1_bekko/stage1.pt \
        --judgements data/judgements/open-jev-validation.jsonl

Read the table for the point where recall@1 clears its floor while the fidelity metrics
still clear theirs, if one exists. If none does, that is the finding, and the level-2 recipe
is wrong rather than this model's floors.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy

import torch

from byteify.contrastive import retrieval_over_judgements, train_contrastive_on_corpus
from byteify.data import truncate_to_bytes
from byteify.fidelity import _budgeted_patch_reps, measure_fidelity
from byteify.gates import assets_path, thresholds_for
from byteify.hub import resolve_source_model, resolve_source_tokenizer
from byteify.inference import ByteifiedEmbedder
from byteify.source import (
    HuggingFaceSourceEncoder,
    LocalEncoderSpec,
    build_byteified_encoder,
    load_encoder_state,
)
from byteify.stitching import encode_texts
from byteify.training import load_judgements, load_reference_texts

#: The trade-off points: (contrastive steps, learning rate).
LADDER = ((0, 0.0), (1500, 1e-4), (500, 1e-4), (150, 1e-4), (1500, 2e-5), (500, 2e-5), (150, 2e-5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--judgements", required=True)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--local-ffn", type=int, default=1024)
    parser.add_argument("--out", default="runs/contrastive_tradeoff.json")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    thresholds = thresholds_for(args.model)
    tokenizer = resolve_source_tokenizer(args.model)
    source = HuggingFaceSourceEncoder(resolve_source_model(args.model).model, model_id=args.model)
    source = source.to(device).eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    encoder = build_byteified_encoder(
        source, tokenizer, LocalEncoderSpec(layers=1, ffn_hidden=args.local_ffn, window=128)
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["encoder"]
    load_encoder_state(encoder, state, source=str(args.checkpoint))
    stage1_state = deepcopy(encoder.state_dict())
    encoder.embed.load_suffix_from(source.embedding_matrix())

    texts = [truncate_to_bytes(t, args.max_bytes) for t in load_reference_texts(args.data)]
    eval_texts = texts[: args.n_eval]
    batches = [
        encode_texts(eval_texts[i : i + args.batch_size], tokenizer, device=device)
        for i in range(0, len(eval_texts) - args.batch_size + 1, args.batch_size)
    ]
    judgements = load_judgements(args.judgements)

    print(f"model     {args.model}")
    print(f"floors    cos_emb_c {thresholds.cos_emb_c}, sim_corr {thresholds.sim_corr}, "
          f"boundary_acc {thresholds.boundary_acc}, "
          f"retrieval_recall_at_1 {thresholds.retrieval_recall_at_1}")
    print(f"evaluation {len(eval_texts)} texts / {len(judgements)} decisions, "
          f"byte axis {args.max_bytes}, budget {args.budget}\n")
    header = f"{'steps':>6} {'lr':>8} {'cos_emb_c':>10} {'sim_corr':>9} {'boundary':>9} {'recall@1':>9} {'l1':>4}"
    print(header)
    print("-" * len(header))

    rows = []
    for steps, lr in LADDER:
        encoder.load_state_dict(stage1_state)
        if steps:
            train_contrastive_on_corpus(
                encoder, source, tokenizer, texts,
                steps=steps, lr=lr, max_bytes=args.max_bytes, seed=0, device=device,
                log=lambda message: None,
            )
        encoder.eval()
        embedder = ByteifiedEmbedder(
            encoder, source, n_patches=min(args.budget, args.max_bytes)
        ).to(device).eval()

        def fidelity_reps(batch, _embedder=embedder):
            return _budgeted_patch_reps(_embedder, batch)

        def embed_batch(texts, _embedder=embedder):
            return _embedder.encode(tokenizer, texts, max_bytes=args.max_bytes)

        fidelity = measure_fidelity(fidelity_reps, source, batches)
        retrieval = retrieval_over_judgements(
            embed_batch,
            judgements,
            batch_size=32,
            max_bytes=args.max_bytes,
        )
        holds = (
            fidelity["cos_emb_c"] >= thresholds.cos_emb_c
            and fidelity["sim_corr"] >= thresholds.sim_corr
            and fidelity["boundary_acc"] >= thresholds.boundary_acc
        )
        reaches = retrieval["recall@1"] >= thresholds.retrieval_recall_at_1
        print(f"{steps:>6} {lr:>8.0e} {fidelity['cos_emb_c']:>10.4f} {fidelity['sim_corr']:>9.4f} "
              f"{fidelity['boundary_acc']:>9.4f} {retrieval['recall@1']:>9.4f} "
              f"{'yes' if holds else 'NO':>4}")
        rows.append({
            "steps": steps, "lr": lr,
            **{k: fidelity[k] for k in ("cos_emb_c", "sim_corr", "boundary_acc")},
            "recall@1": retrieval["recall@1"],
            "mrr": retrieval["mrr"],
            "level1_holds": holds, "retrieval_floor_reached": reaches,
            "both": holds and reaches,
        })

    feasible = [row for row in rows if row["both"]]
    print()
    if feasible:
        best = max(feasible, key=lambda row: row["recall@1"])
        print(f"A point exists: {best['steps']} steps at lr {best['lr']:.0e} clears the level-1 "
              f"floors and recall@1 {best['recall@1']:.4f}.")
    else:
        print("No point on this ladder clears the level-1 floors and the retrieval floor "
              "together. Either the contrastive recipe moves the byte interface further than "
              f"level 1 can tolerate, or the floors in {assets_path('thresholds.json')} do not "
              "leave room for a level-2 artifact.")
    print(f"chance recall@1 on these judgements: {judgements.chance_recall_at_1:.4f}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "ladder": rows}, handle, indent=2, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()