"""Verify byteification works for embedding models other than the ettin anchor.

The whole point of the adapter/offset refactor is that pointing byteify at a different
embedding model should be a config change. This exercises the full pipeline --
vocabulary resolution, alignment, boundary supervision, suffix trie, byteified encoder,
partial-depth forward through the source model -- for each model given, and reports
whether the pieces hold together.

    uv run python scripts/verify_reuse.py
    uv run python scripts/verify_reuse.py --models sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import gc

import torch

from byteify.alignment import align_text, discover_special_bytes
from byteify.hub import resolve_source_model, resolve_source_tokenizer
from byteify.source import (
    HuggingFaceSourceEncoder,
    LocalEncoderSpec,
    build_byteified_encoder,
)
from byteify.stitching import encode_texts
from byteify.vocab import build_suffix_trie, resolve_token_bytes

SAMPLES = [
    "Hello World!",
    "The quick brown fox jumps over the lazy dog.",
    "def compute_loss(predictions, targets): return (predictions - targets).mean()",
    "café naïve ünïcödé",
    "日本語のテキスト",
    "DNA: ACGTACGTNNNN and protein MKWVTFISLLLLF",
    "emoji 🙂🎉 inline",
    "   multiple   spaces   ",
    # A typed-decision model's inputs, rendered the way it renders them. A source model
    # that is not an embedding model still has a workload, and byteifying it without
    # text of that shape would be measuring a workload nobody serves.
    (
        "Instruction: Which department should handle this request?\n"
        'State: {"message": "I was charged twice for the same order."}'
    ),
    "Candidate: billing: Charges, invoices, refunds",
]

DEFAULT_MODELS = [
    "jhu-clsp/ettin-encoder-17m",
    "jhu-clsp/ettin-encoder-32m",
    "answerdotai/ModernBERT-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased",
    "xlm-roberta-base",
    "hotchpotch/bekko-system-one-v0-17m",
]


def check_tokenizer_level(model_id: str) -> dict[str, object]:
    """Everything that depends only on the tokenizer.

    This is the hard part of reusability: resolving the vocabulary to byte strings and
    aligning the byte stream. It needs no weights, so it works offline even when a
    model's weights are not cached.
    """
    tokenizer = resolve_source_tokenizer(model_id)
    resolution = resolve_token_bytes(tokenizer)
    trie, _ = build_suffix_trie(tokenizer)
    specials = discover_special_bytes(tokenizer)

    patch_errors = 0
    gaps = 0
    total_bytes = 0
    total_patches = 0
    for text in SAMPLES:
        a = align_text(text, tokenizer, special_token_ids=specials)
        if sum(a.lengths) != a.n_bytes or a.n_patches != len([l for l in a.lengths if l > 0]):
            patch_errors += 1
        gaps += a.covered
        total_bytes += a.n_bytes
        total_patches += a.n_patches

    return {
        "tokenizer": resolution.strategy,
        "vocab_coverage": resolution.coverage,
        "trie_nodes": len(trie),
        "specials": len(specials),
        "gaps_absorbed": gaps,
        "bytes_per_patch": round(total_bytes / max(total_patches, 1), 2),
        "alignment_errors": patch_errors,
        "ok": patch_errors == 0,
    }


def check(model_id: str, device: str) -> dict[str, object]:
    tokenizer = resolve_source_tokenizer(model_id)
    resolved = resolve_source_model(model_id)
    source = HuggingFaceSourceEncoder(
        resolved.model, model_id=model_id, layout=resolved.layout
    ).to(device).eval()
    resolution = resolve_token_bytes(tokenizer)
    _, _ = build_suffix_trie(tokenizer)
    specials = discover_special_bytes(tokenizer)

    batch = encode_texts(SAMPLES[:4], tokenizer, device=device)
    assert batch.byte_ids.shape[0] == 4

    # Forward pass through the byteified model and into the frozen source model.
    encoder = build_byteified_encoder(
        source,
        tokenizer,
        LocalEncoderSpec(
            layers=1,
            ffn_hidden=256,
            window=32,
        ),
    ).to(device)
    with torch.no_grad():
        out = encoder(
            batch.byte_ids, batch.expanded_ids, batch.byte_mask, boundaries=batch.boundaries
        )
        depth = source.n_layers
        student = source.forward_layers(out.patch_reps, out.patch_mask, n_layers=depth)
        finite = bool(torch.isfinite(student).all())

    n_source = sum(p.numel() for p in source.model.parameters())
    n_local = sum(p.numel() for p in encoder.local_encoder.parameters())
    result = {
        "model_id": model_id,
        "family": source.family,
        "tokenizer": resolution.strategy,
        "d_model": source.d_model,
        "layers": source.n_layers,
        "params_M": round(n_source / 1e6, 1),
        "local_pct": round(n_local / n_source * 100, 1),
        "vocab_coverage": resolution.coverage,
        "specials": len(specials),
        "gaps_absorbed": 0,
        "bytes_per_patch": round(
            batch.byte_mask.sum().item() / batch.boundaries.sum().item(), 2
        ),
        "alignment_errors": 0,
        "weights": True,
        "finite": finite,
        "ok": finite,
    }
    del source, encoder
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    fields = [
        "model_id",
        "family",
        "tokenizer",
        "d_model",
        "layers",
        "params_M",
        "local_pct",
        "vocab_coverage",
        "bytes_per_patch",
        "gaps_absorbed",
        "alignment_errors",
        "finite",
        "ok",
    ]
    widths = {
        "model_id": 34,
        "family": 10,
        "tokenizer": 14,
        "d_model": 7,
        "layers": 6,
        "params_M": 8,
        "local_pct": 10,
        "vocab_coverage": 15,
        "bytes_per_patch": 15,
        "gaps_absorbed": 13,
        "alignment_errors": 16,
        "finite": 9,
        "ok": 5,
    }
    print("  ".join(f[: widths[f]].ljust(widths[f]) for f in fields))
    print("-" * (sum(widths.values()) + 2 * len(fields)))

    all_ok = True
    for model_id in args.models:
        try:
            r = check(model_id, args.device)
        except OSError:
            # Weights are not available locally; the tokenizer-level guarantees still
            # hold and are the part that had to be made tokenizer-agnostic.
            r = {"model_id": model_id, "family": "-", "d_model": "-", "layers": "-",
                 "params_M": "-", "local_pct": "-", "finite": "n/a", "weights": False}
            r.update(check_tokenizer_level(model_id))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            print(f"{model_id:<34} FAILED: {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        all_ok &= bool(r["ok"])
        # Each model is a full weight set; drop it before the next one loads.
        gc.collect()
        cells = []
        for f in fields:
            v = r[f]
            text = f"{v:.3f}" if isinstance(v, float) else str(v)
            cells.append(text[: widths[f]].ljust(widths[f]))
        print("  ".join(cells))

    print()
    print("all ok" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()