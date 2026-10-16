import pytest
import torch

from byteify.model import (
    ByteifiedEncoder,
    CosineBoundaryPredictor,
    LocalContextEncoder,
    MLPBoundaryPredictor,
    PartialModernBert,
    patch_index,
    pool_patches,
)

torch.manual_seed(0)


def test_patch_index_counts_patches():
    # boundaries at 2 and 4 -> patch 0 owns bytes 0-2, patch 1 owns bytes 3-4.
    boundaries = torch.tensor([[0, 0, 1, 0, 1]])
    assert patch_index(boundaries).tolist() == [[0, 0, 0, 0, 1]]


def test_pool_patches_takes_last_byte_of_each_patch():
    h = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    boundaries = torch.tensor([[0, 1, 0, 1, 0, 1]])
    reps, mask = pool_patches(h, boundaries)
    assert mask.tolist() == [[True, True, True]]
    # patches are bytes (0,1), (2,3), (4,5) -> last byte of each
    assert reps[0, 0].tolist() == [2.0, 3.0]
    assert reps[0, 1].tolist() == [6.0, 7.0]
    assert reps[0, 2].tolist() == [10.0, 11.0]


def test_pool_patches_ignores_padding():
    """Padded positions must not overwrite the final real patch.

    Padding carries the same patch index as the sequence's last patch, so a naive
    scatter silently corrupts the pooled vector of the longest sequence in a batch.
    """
    # Two sequences; the second is padded from 3 to 5 bytes.
    h = torch.zeros(2, 5, 2)
    h[0, :4] = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    h[1, :3] = torch.tensor([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
    boundaries = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 1, 0, 0]])

    reps, mask = pool_patches(h, boundaries)
    assert mask.tolist() == [[True, True], [True, True]]
    assert reps[0, 0].tolist() == [2.0, 2.0]
    assert reps[0, 1].tolist() == [4.0, 4.0]
    assert reps[1, 0].tolist() == [6.0, 6.0]
    assert reps[1, 1].tolist() == [7.0, 7.0], "padding must not clobber the last real patch"


def test_pool_patches_is_permutation_of_ends():
    torch.manual_seed(1)
    h = torch.randn(3, 9, 4)
    boundaries = (torch.rand(3, 9) < 0.4)
    boundaries[:, -1] = True
    reps, mask = pool_patches(h, boundaries)
    assert reps.shape == (3, int(boundaries.sum(1).max()), 4)
    assert mask.sum(1).tolist() == boundaries.sum(1).tolist()
    for b in range(3):
        ends = boundaries[b].nonzero().flatten()
        for p, end in enumerate(ends):
            assert torch.equal(reps[b, p], h[b, end])


def test_local_encoder_is_bidirectional():
    """A wide window lets an early position react to a late byte.

    Bolmo's mLSTM local encoder is causal; the byteified encoder's is not. If this
    ever becomes one-directional we lose the ability to place boundaries with
    future context, which is the whole point of the encoder setting.
    """
    enc = LocalContextEncoder(16, n_layers=1, ffn_hidden=32, n_heads=4, window=64)
    enc.eval()
    x = torch.randn(1, 20, 16)
    mask = torch.ones(1, 20, dtype=torch.long)
    a = enc(x, mask)
    x2 = x.clone()
    x2[:, -1] += 10.0
    b = enc(x2, mask)
    assert not torch.allclose(a[:, 0], b[:, 0]), "position 0 must react to the final byte"


def test_local_encoder_is_local():
    """The window keeps cost linear: positions beyond the window do not interact."""
    enc = LocalContextEncoder(16, n_layers=1, ffn_hidden=32, n_heads=4, window=4)
    enc.eval()
    x = torch.randn(1, 20, 16)
    mask = torch.ones(1, 20, dtype=torch.long)
    a, x2 = enc(x, mask), None
    x2 = x.clone()
    x2[:, -1] += 10.0
    b = enc(x2, mask)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-6), "out-of-window position must not change"
    assert not torch.allclose(a[:, 15], b[:, 15]), "in-window position must change"


def test_mlp_boundary_head_is_position_wise():
    """Context comes from the local encoder, not the readout head.

    Keeping the head position-wise means all of the boundary predictor's
    non-causality is supplied by the local encoder, which is the component we
    control separately. See test_boundary_logits_use_full_context for the
    end-to-end property that matters.
    """
    pred = MLPBoundaryPredictor(8, 16)
    pred.eval()
    x = torch.zeros(1, 6, 8)
    a = pred(x)
    x2 = x.clone()
    x2[:, -1] = 1.0
    b = pred(x2)
    assert torch.allclose(a[:, 0], b[:, 0]), "head alone must be position-wise"


def test_boundary_logits_use_full_context():
    """End to end, an early byte's boundary score may depend on later bytes.

    This is the property that lets the boundary predictor decide "flower" is a
    token only once it sees "bed" -- the subword tokenizer's non-causality that
    Bolmo could only approximate with 1 byte of lookahead.
    """
    enc = ByteifiedEncoder(
        d_model=16,
        n_byte_values=260,
        subword_vocab_size=10,
        local_encoder=LocalContextEncoder(16, n_layers=1, ffn_hidden=32, n_heads=4, window=64),
        boundary_predictor="mlp",
    )
    enc.eval()
    byte_ids = torch.zeros(1, 20, dtype=torch.long)
    expanded = torch.zeros(1, 20, dtype=torch.long)
    mask = torch.ones(1, 20, dtype=torch.long)

    a = enc(byte_ids, expanded, mask, boundaries=torch.ones(1, 20, dtype=torch.long))
    byte_ids2 = byte_ids.clone()
    byte_ids2[:, -1] = 200
    b = enc(byte_ids2, expanded, mask, boundaries=torch.ones(1, 20, dtype=torch.long))
    assert not torch.allclose(
        a.boundary_logits[:, 0], b.boundary_logits[:, 0]
    ), "boundary score at position 0 must see the final byte"


def test_local_encoder_respects_padding():
    enc = LocalContextEncoder(16, n_layers=1, ffn_hidden=32, n_heads=4, window=8)
    enc.eval()
    x = torch.randn(1, 12, 16)
    mask = torch.ones(1, 12, dtype=torch.long)
    mask[:, 6:] = 0
    a = enc(x, mask)
    x2 = x.clone()
    x2[:, 6:] = 100.0  # garbage in the padded region
    b = enc(x2, mask)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5), "padding leaked into real positions"


def test_local_encoder_gru_backend():
    enc = LocalContextEncoder(16, n_layers=1, backend="gru")
    out = enc(torch.randn(2, 10, 16), torch.ones(2, 10, dtype=torch.long))
    assert out.shape == (2, 10, 16)


def test_cosine_boundary_predictor_peaks_at_discontinuities():
    """The score is high where consecutive bytes are unlike each other.

    ``score = 0.5 * (1 - cos)``, so identical neighbours score 0 and the score only
    exceeds 0.5 once the cosine actually goes negative.
    """
    pred = CosineBoundaryPredictor(8, lookahead=1)
    u = torch.tensor([1.0] + [0.0] * 7)
    v = torch.tensor([-0.5, 1.0] + [0.0] * 6)
    reps = torch.zeros(1, 10, 8)
    reps[:, 0:4] = u
    reps[:, 4:8] = v
    reps[:, 8:10] = u

    scores = pred(reps)[0]
    assert scores.shape == (10,)
    assert scores[3] > 0.5, "expected a boundary at the u -> v transition"
    assert scores[7] > 0.5, "expected a boundary at the v -> u transition"
    for i in (0, 1, 2, 4, 5, 6, 8):
        assert scores[i] < 1e-5, f"position {i} is inside a run and must score ~0"


def test_byteified_encoder_forward_shapes():
    enc = ByteifiedEncoder(
        d_model=16,
        n_byte_values=260,
        subword_vocab_size=100,
        local_encoder=LocalContextEncoder(16, n_layers=1, ffn_hidden=32, n_heads=4, window=8),
    )
    byte_ids = torch.randint(0, 256, (2, 20))
    expanded = torch.randint(0, 100, (2, 20))
    byte_mask = torch.ones(2, 20, dtype=torch.long)
    boundaries = torch.zeros(2, 20, dtype=torch.long)
    boundaries[:, 3] = boundaries[:, 9] = boundaries[:, 19] = 1

    out = enc(byte_ids, expanded, byte_mask, boundaries)
    assert out.byte_reps.shape == (2, 20, 16)
    assert out.patch_reps.shape == (2, 3, 16)
    assert out.patch_mask.shape == (2, 3)
    assert out.boundary_logits is not None
    assert out.boundary_logits.shape == (2, 20)


def test_suffix_embedding_ignores_missing_expansions():
    enc = ByteifiedEncoder(d_model=8, n_byte_values=260, subword_vocab_size=50)
    byte_ids = torch.zeros(1, 4, dtype=torch.long)
    expanded = torch.tensor([[-1, 3, -1, 7]])
    e = enc.embed(byte_ids, expanded)
    # Positions with expansion -1 get byte embedding only.
    assert torch.equal(e[0, 0], enc.embed.byte.weight[0])
    assert torch.equal(e[0, 1], enc.embed.byte.weight[0] + enc.embed.suffix.weight[3])


def test_partial_modernbert_matches_hidden_states():
    pytest.importorskip("transformers")
    from transformers import AutoModel

    model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-17m")
    model.eval()
    partial = PartialModernBert(model)

    ids = torch.tensor([[50281, 12092, 64, 14947, 2, 50282]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states

        for n in (0, 1, 4):
            # Pass pre-norm embeddings: PartialModernBert applies the embedding norm
            # itself, exactly as inputs_embeds would on the student's patch reps.
            got = partial(model.embeddings.tok_embeddings(ids), mask, n_layers=n)
            if n == 0:
                assert torch.allclose(got, ref[0], atol=1e-5)
            else:
                assert torch.allclose(got, ref[n], atol=1e-5), f"layer {n} mismatch"

    assert partial.n_layers == 7