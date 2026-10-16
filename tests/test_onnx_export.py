"""ONNX export: the graph must exist, load, and match eager PyTorch numerically."""

import numpy as np
import pytest
import torch

from byteify.inference import ByteifiedEmbedder, export_onnx

MODEL = "jhu-clsp/ettin-encoder-17m"


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("transformers")
    from transformers import AutoModel, AutoTokenizer

    from byteify.alignment import discover_special_bytes
    from byteify.model import ByteifiedEncoder, LocalContextEncoder
    from byteify.source import HuggingFaceSourceEncoder
    from byteify.vocab import resolve_token_bytes

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained(MODEL)).eval()
    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    return tokenizer, ByteifiedEmbedder(encoder, source, n_patches=8)


def _inputs(tokenizer, max_bytes, batch=2):
    from byteify.inference import byte_ids_for_inference

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Byte-level embeddings avoid a tokenizer.",
    ][:batch]
    rows = [byte_ids_for_inference(tokenizer, t, max_bytes) for t in texts]
    width = max_bytes
    pad = lambda ts, v: torch.stack(
        [torch.nn.functional.pad(t.reshape(-1), (0, width - t.shape[-1]), value=v) for t in ts]
    )
    return pad([r[0] for r in rows], 0), pad([r[1] for r in rows], -1), pad([r[2] for r in rows], 0)


def test_export_produces_a_loadable_graph(embedder, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    tokenizer, embedder = embedder
    byte_ids, expanded, mask = _inputs(tokenizer, 64)
    path = str(tmp_path / "m.onnx")
    export_onnx(
        embedder,
        tokenizer,
        path=path,
        n_patches=8,
        example_batch=(byte_ids, expanded, mask),
    )

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    assert names == ["byte_ids", "expanded_ids", "byte_mask"]
    assert len(session.get_outputs()) == 1


def test_onnx_matches_eager(embedder, tmp_path):
    """The whole point: an artifact that loads but computes something else is useless."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    tokenizer, embedder = embedder
    byte_ids, expanded, mask = _inputs(tokenizer, 64)
    path = str(tmp_path / "m.onnx")
    export_onnx(
        embedder,
        tokenizer,
        path=path,
        n_patches=8,
        example_batch=(byte_ids, expanded, mask),
    )

    with torch.no_grad():
        eager = embedder(byte_ids, expanded, mask)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(
        None,
        {
            "byte_ids": byte_ids.numpy(),
            "expanded_ids": expanded.numpy(),
            "byte_mask": mask.numpy(),
        },
    )[0]

    assert got.shape == eager.shape
    assert np.abs(got - eager.numpy()).max() < 1e-4
    cos = torch.nn.functional.cosine_similarity(torch.from_numpy(got), eager, dim=-1)
    assert cos.min().item() > 0.9999


def test_onnx_accepts_a_different_batch_size(embedder, tmp_path):
    """Batch is the dynamic axis; a different batch size must still run."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    tokenizer, embedder = embedder
    _, _, _ = _inputs(tokenizer, 64, batch=2)
    path = str(tmp_path / "m.onnx")
    export_onnx(embedder, tokenizer, path=path, n_patches=8, example_batch=_inputs(tokenizer, 64, batch=2))

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    big = _inputs(tokenizer, 64, batch=1)
    replicated = torch.cat([big[0], big[0]], dim=0)
    out = session.run(
        None,
        {
            "byte_ids": replicated.numpy(),
            "expanded_ids": torch.cat([big[1], big[1]], dim=0).numpy(),
            "byte_mask": torch.cat([big[2], big[2]], dim=0).numpy(),
        },
    )[0]
    assert out.shape == (2, embedder.source.d_model)


def test_export_rejects_mismatched_input_widths(embedder, tmp_path):
    """A mask shorter than the ids makes the encoder see a 4-D tensor.

    This is the failure mode that produced a long, misleading debugging detour: the
    export died inside LocalAttention reporting a 4-D activation rather than at the
    point where the inputs actually diverged.
    """
    pytest.importorskip("onnxscript")
    tokenizer, embedder = embedder
    byte_ids, expanded, mask = _inputs(tokenizer, 64)
    with pytest.raises(ValueError, match="share a byte axis"):
        export_onnx(
            embedder,
            tokenizer,
            path=str(tmp_path / "bad.onnx"),
            n_patches=8,
            example_batch=(byte_ids, expanded, mask[:, :-8]),
        )