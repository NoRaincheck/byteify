"""Inference path: budgeted patching, pooling, and ONNX export."""

import pytest
import torch

from byteify.inference import (
    ByteifiedEmbedder,
    budgeted_patches_mask,
    pool_budgeted_patches,
    select_budgeted_patches,
)
from byteify.model import ByteifiedEncoder, LocalContextEncoder

torch.manual_seed(0)


def test_budgeted_patches_have_static_length():
    """The whole point: the patch axis cannot depend on the input."""
    probs = torch.rand(3, 64)
    mask = torch.ones(3, 64, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=8)
    assert out.end_indices.shape == (3, 8)
    assert out.mask.shape == (3, 8)


def test_budgeted_patches_are_ordered_and_unique():
    probs = torch.rand(4, 50)
    mask = torch.ones(4, 50, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=7)
    for row in out.end_indices:
        assert row.tolist() == sorted(row.tolist())
        assert len(set(row.tolist())) == 7


def test_budgeted_patches_never_select_padding():
    probs = torch.rand(2, 40)
    mask = torch.zeros(2, 40, dtype=torch.long)
    mask[:, :10] = 1
    out = select_budgeted_patches(probs, mask, n_patches=4)
    assert (out.end_indices < 10).all(), "a patch end landed in the padded region"


def test_budgeted_patches_terminate_the_sequence():
    """The last real byte must be a patch end or trailing text is dropped."""
    probs = torch.rand(2, 30) * 0.1
    mask = torch.zeros(2, 30, dtype=torch.long)
    mask[:, :12] = 1
    out = select_budgeted_patches(probs, mask, n_patches=5, threshold=0.5)
    assert 11 in out.end_indices[0].tolist()
    assert 11 in out.end_indices[1].tolist()
    assert bool(out.mask[0, (out.end_indices[0] == 11).nonzero()[0, 0]])


def test_budgeted_patches_mask_low_confidence_slots():
    """Few confident boundaries -> surplus slots masked out of pooling, not fabricated."""
    probs = torch.full((1, 40), 0.01)
    probs[0, 5] = 0.99
    probs[0, 39] = 0.99
    mask = torch.ones(1, 40, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=8, threshold=0.5)
    assert out.mask.sum().item() == 2, "only the two confident boundaries are real patches"


def test_budgeted_patches_keeps_topk_when_over_budget():
    """More confident boundaries than budget -> the least confident ones are dropped."""
    probs = torch.linspace(0.9, 0.99, 20).unsqueeze(0)
    probs[0, 19] = 2.0
    mask = torch.ones(1, 20, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=5, threshold=0.5)
    assert out.end_indices.shape == (1, 5)
    # The highest-probability positions survive.
    assert out.mask.all(), "all selected boundaries were above threshold"


def test_budgeted_patches_rejects_oversized_budget():
    with pytest.raises(ValueError, match="exceeds the byte axis"):
        select_budgeted_patches(torch.rand(1, 10), torch.ones(1, 10, dtype=torch.long), n_patches=11)


def test_pool_budgeted_patches_takes_last_byte_and_masks():
    h = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
    probs = torch.rand(1, 10)
    mask = torch.ones(1, 10, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=3, threshold=0.0)
    reps = pool_budgeted_patches(h, out)
    assert reps.shape == (1, 3, 4)
    for i in range(3):
        end = int(out.end_indices[0, i])
        if bool(out.mask[0, i]):
            assert torch.allclose(reps[0, i], h[0, end])


def test_pool_budgeted_patches_zeroes_masked_slots():
    h = torch.randn(1, 10, 4)
    probs = torch.zeros(1, 10)
    probs[0, 9] = 0.9
    mask = torch.ones(1, 10, dtype=torch.long)
    out = select_budgeted_patches(probs, mask, n_patches=4, threshold=0.5)
    reps = pool_budgeted_patches(h, out)
    assert reps[0, ~out.mask[0]].abs().sum() == 0


def test_embedder_forward_shape_and_finiteness():
    pytest.importorskip("transformers")
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained("jhu-clsp/ettin-encoder-17m")).eval()
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=260,
        subword_vocab_size=50368,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    embedder = ByteifiedEmbedder(encoder, source, n_patches=8).eval()

    byte_ids = torch.randint(0, 256, (2, 40))
    expanded = torch.randint(0, 50368, (2, 40))
    byte_mask = torch.ones(2, 40, dtype=torch.long)
    with torch.no_grad():
        emb = embedder(byte_ids, expanded, byte_mask)
    assert emb.shape == (2, source.d_model)
    assert torch.isfinite(emb).all()
    # Normalised by default.
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_embedder_output_is_deterministic():
    pytest.importorskip("transformers")
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained("jhu-clsp/ettin-encoder-17m")).eval()
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=260,
        subword_vocab_size=50368,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    embedder = ByteifiedEmbedder(encoder, source, n_patches=8).eval()
    byte_ids = torch.randint(0, 256, (1, 32))
    expanded = torch.randint(0, 50368, (1, 32))
    byte_mask = torch.ones(1, 32, dtype=torch.long)
    with torch.no_grad():
        a = embedder(byte_ids, expanded, byte_mask)
        b = embedder(byte_ids, expanded, byte_mask)
    assert torch.equal(a, b)


def test_embedder_distinguishes_inputs():
    """A model that returned one constant vector would pass every other test here."""
    pytest.importorskip("transformers")
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    source = HuggingFaceSourceEncoder(AutoModel.from_pretrained("jhu-clsp/ettin-encoder-17m")).eval()
    torch.manual_seed(0)
    encoder = ByteifiedEncoder(
        d_model=source.d_model,
        n_byte_values=260,
        subword_vocab_size=50368,
        local_encoder=LocalContextEncoder(
            source.d_model, n_layers=1, ffn_hidden=64, n_heads=4, window=16
        ),
    ).eval()
    embedder = ByteifiedEmbedder(encoder, source, n_patches=8).eval()
    byte_ids = torch.randint(0, 256, (2, 40))
    expanded = torch.randint(0, 50368, (2, 40))
    byte_mask = torch.ones(2, 40, dtype=torch.long)
    byte_ids[1] = (byte_ids[0] + 97) % 256
    with torch.no_grad():
        emb = embedder(byte_ids, expanded, byte_mask)
    assert not torch.allclose(emb[0], emb[1], atol=1e-4)


def test_budgeted_patches_mask_helper():
    """Convenience helper used by tests and by the export wrapper."""
    mask = torch.tensor([[True, False, True]])
    assert budgeted_patches_mask(mask).tolist() == [[1, 0, 1]]