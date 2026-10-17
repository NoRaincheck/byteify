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
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from byteify.inference import ByteifiedEmbedder
from byteify.manifest import export_artifact
from byteify.source import HuggingFaceSourceEncoder, load_byteified_encoder
from byteify.vocab import resolve_token_bytes

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Byte-level embeddings avoid a tokenizer entirely.",
    "def add(a, b): return a + b",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/ettin-encoder-17m")
    p.add_argument("--init", default=None, help="trained checkpoint to export")
    p.add_argument("--out", default="artifact.onnx", help="graph file name")
    p.add_argument("--out-dir", default=None, help="artifact directory (default: --out's)")
    p.add_argument("--max-bytes", type=int, default=512)
    p.add_argument("--n-patches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-4)
    return p.parse_args()


def build(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = HuggingFaceSourceEncoder(
        AutoModel.from_pretrained(args.model), model_id=args.model
    ).to(device).eval()

    # A checkpoint is required. Exporting a randomly initialised interface produces a
    # graph that loads and matches eager PyTorch perfectly, which is precisely why the
    # fidelity gate exists and why this script will not make the artifact for you.
    if not args.init:
        raise SystemExit(
            "--init is required: an artifact must be built from trained weights, not "
            "from a randomly initialised byte interface"
        )
    encoder = load_byteified_encoder(args.init, source, tokenizer).to(device).eval()
    print(f"loaded {args.init}")

    # The same rule and trie the artifact will carry, so the graph is verified against
    # the inputs a serving host will actually hand it.
    from byteify.alignment import discover_special_bytes
    from byteify.bytestream import rule_for_family
    from byteify.vocab import build_suffix_trie

    specials = discover_special_bytes(tokenizer)
    rule = rule_for_family(
        resolve_token_bytes(tokenizer).strategy,
        n_byte_values=256 + len(specials),
        special_tokens=specials,
    )
    trie, _ = build_suffix_trie(tokenizer)
    return tokenizer, source, encoder, rule, trie


def _inputs_from_rule(rule, trie, text: str, max_bytes: int):
    """Build graph inputs from the declared byte-stream rule, not from the tokenizer.

    This is the path a serving host takes. Verifying the graph against inputs built this
    way is the point: verifying against tokenizer-produced inputs would only prove the
    graph agrees with a path it will never be given.
    """
    from byteify.expand import expand_byte_ids

    byte_ids = rule.build(text, max_bytes)
    expanded = expand_byte_ids(byte_ids, trie, pseudo_byte_to_token=rule.pseudo_bytes)
    return (
        torch.tensor([byte_ids], dtype=torch.long),
        torch.tensor([expanded], dtype=torch.long),
        torch.ones(1, len(byte_ids), dtype=torch.long),
    )


def main():
    args = parse_args()
    device = args.device
    tokenizer, source, encoder, rule, trie = build(args, device)
    embedder = ByteifiedEmbedder(encoder, source, n_patches=args.n_patches).to(device).eval()

    # Pad a batch to a common byte length, as a real serving path would.
    # Pad to a fixed byte length: the exported graph fixes the byte axis, and
    # padding is what a serving path does to batch requests anyway.
    width = args.max_bytes

    def pad_to_width(tensor):
        return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]))

    prepared = [
        _inputs_from_rule(rule, trie, t, args.max_bytes)
        for t in SAMPLES[: args.batch_size]
    ]

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

    # One call writes the graph, the trie, the encoder weights and the manifest, and
    # refuses to write a manifest that does not describe what it just exported. The
    # manifest goes last: until it exists the directory is not an artifact.
    artifact_dir = Path(args.out_dir) if args.out_dir else Path(args.out).parent
    manifest = export_artifact(
        embedder,
        tokenizer,
        checkpoint=Path(args.init),
        output_dir=artifact_dir,
        budget=args.n_patches,
        max_bytes=args.max_bytes,
        coverage_texts=SAMPLES,
        graph_name=Path(args.out).name,
        example_batch=(byte_ids, expanded, byte_mask),
    )
    path = artifact_dir / Path(args.out).name
    size_mb = path.stat().st_size / 1e6
    print(f"exported {path} ({size_mb:.1f} MB)")
    print(f"wrote {artifact_dir / 'manifest.json'}")
    print(f"  source      {manifest.source_id} ({manifest.architecture})")
    print(f"  family      {manifest.family}, byte stream {manifest.byte_stream_rule}")
    print(f"  byte axis   {manifest.max_bytes}, patch budget {manifest.budget}")
    print(f"  byte table  {manifest.n_byte_values} values, vocab {manifest.vocab_size}")
    if manifest.trie.coverage is not None:
        print(f"  trie        {manifest.trie.entries} entries, "
              f"{manifest.trie.coverage * 100:.1f}% coverage over "
              f"{manifest.trie.coverage_bytes} bytes")

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