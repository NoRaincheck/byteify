"""Export a trained byteified embedder to ONNX, and verify it numerically.

    uv run python scripts/export_onnx.py --init runs/contrastive/contrastive.pt

The exported graph takes ``byte_ids``, ``expanded_ids`` and ``byte_mask`` and returns a
sentence embedding. It contains the whole path: byte embedding, local encoder, boundary
prediction, budgeted patching, the frozen global model, output norm, mean pool.

Two design decisions make the export possible, both in ``byteify.inference``:

* **Budgeted patching** fixes the patch axis to a constant, since ONNX needs static
  shapes and the patch count would otherwise depend on the input.
* **No data-dependent Python control flow.** ONNX cannot branch on tensor values, so
  the patch budget is threaded as a tensor.

After export this script re-runs the ONNX graph and compares against eager PyTorch. If
the two disagree, the artifact is not trustworthy regardless of what the file loads as.
"""

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.inference import ByteifiedEmbedder, byte_ids_for_inference, export_onnx
from byteify.source import HuggingFaceSourceEncoder, load_byteified_encoder

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Byte-level embeddings avoid a tokenizer entirely.",
    "def add(a, b): return a + b",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--init", default=None, help="trained checkpoint to export")
    p.add_argument("--out", default="byteified_embedder.onnx")
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--n-patches", type=int, default=64)
    p.add_argument("--opset", type=int, default=18,
                   help="minimum 18: the dynamo exporter requires it")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-4)
    return p.parse_args()


def build(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(args.model)).to(device).eval()

    if args.init:
        # The architecture comes out of the checkpoint. Rebuilding it from constants here
        # produced a different shape for any checkpoint not trained with those constants.
        encoder = load_byteified_encoder(args.init, source, tokenizer)
        encoder = encoder.to(device)
        print(f"loaded {args.init}")
    else:
        from byteify.source import build_byteified_encoder

        encoder = build_byteified_encoder(source, tokenizer)
        encoder.embed.load_suffix_from(source.embedding_matrix())
        print("no --init given: exporting a randomly initialised local encoder")
    encoder = encoder.eval()
    return tokenizer, source, encoder


def main():
    args = parse_args()
    device = args.device
    tokenizer, source, encoder = build(args, device)
    embedder = ByteifiedEmbedder(encoder, source, n_patches=args.n_patches).to(device).eval()

    # Pad a batch to a common byte length, as a real serving path would.
    # Pad to a fixed byte length: the exported graph fixes the byte axis, and
    # padding is what a serving path does to batch requests anyway.
    width = args.max_bytes

    def pad_to_width(tensor):
        return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]))

    prepared = [byte_ids_for_inference(tokenizer, t, args.max_bytes) for t in SAMPLES[: args.batch_size]]

    def pad(tensors, value):
        # Accept either [1, n] or [n] rows; both are flattened to a single axis first.
        rows = [t.reshape(-1) for t in tensors]
        return torch.stack(
            [torch.nn.functional.pad(t, (0, width - t.shape[0]), value=value) for t in rows]
        )

    # Squeeze the leading row that byte_ids_for_inference adds. Each prepared item is
    # [1, n]; stacking those directly yields [batch, 1, n], a 3-D tensor, and the local
    # encoder then sees a 4-D activation and the export fails naming the wrong axis.
    byte_ids = pad([p[0] for p in prepared], 0)
    expanded = pad([p[1] for p in prepared], -1)
    byte_mask = pad([p[2] for p in prepared], 0)
    assert byte_mask.shape == byte_ids.shape == expanded.shape, (
        f"padded shapes disagree: {tuple(byte_ids.shape)} {tuple(expanded.shape)} "
        f"{tuple(byte_mask.shape)}"
    )

    print(f"export shapes: byte_ids={tuple(byte_ids.shape)} "
          f"expanded={tuple(expanded.shape)} mask={tuple(byte_mask.shape)} "
          f"n_patches={args.n_patches}")
    path = export_onnx(
        embedder,
        tokenizer,
        path=args.out,
        max_bytes=args.max_bytes,
        n_patches=args.n_patches,
        opset=args.opset,
        # Export for the exact shape we verify against, so the traced graph and the
        # eager path are known to be comparable.
        example_batch=(byte_ids, expanded, byte_mask),
    )
    size_mb = __import__("os").path.getsize(path) / 1e6
    print(f"exported {path} ({size_mb:.1f} MB)")

    # --- numerical verification -------------------------------------------------
    with torch.no_grad():
        eager = embedder(byte_ids, expanded, byte_mask)

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        print(
            "\nonnxruntime not installed, so the exported graph could not be executed.\n"
            "Install it to verify: uv pip install onnx onnxruntime"
        )
        return

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(
        None, {"byte_ids": byte_ids.numpy(), "expanded_ids": expanded.numpy(), "byte_mask": byte_mask.numpy()}
    )[0]
    eager_np = eager.detach().cpu().numpy()
    diff = float(abs(got - eager_np).max())
    cos = float(
        torch.nn.functional.cosine_similarity(torch.from_numpy(got), eager, dim=-1).mean()
    )

    print(f"\nverification over {len(SAMPLES[: args.batch_size])} texts")
    print(f"  max |onnx - eager| = {diff:.3e}")
    print(f"  mean cosine        = {cos:.6f}")
    if diff > args.atol:
        raise SystemExit(f"FAILED: ONNX output differs by {diff:.3e} (atol {args.atol})")
    print(f"  OK (atol {args.atol})")


if __name__ == "__main__":
    main()