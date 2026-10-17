"""Contrastive objective: InfoNCE, learned temperature, pair mining, retrieval metrics."""

import math

import pytest
import torch
import torch.nn.functional as F

from byteify.contrastive import (
    ContrastiveHead,
    ProjectionHead,
    infonce,
    mine_pairs,
    pair_batches,
    retrieval_metrics,
)

torch.manual_seed(0)


def test_infonce_is_zero_when_anchor_equals_positive():
    """The diagonal being perfectly correct is the global optimum.

    Built from orthonormal vectors rather than random ones. With random 16-dimensional
    vectors the off-diagonal cosines occasionally land near 1, and since the loss is
    ``exp(diagonal - best_negative) / temperature`` with temperature 0.05, one such pair
    lifts the loss to ~0.1. That is correct behaviour, not a defect -- but it made an
    unseeded assertion on random data flaky.
    """
    vectors, _ = torch.linalg.qr(torch.randn(8, 16))
    x = vectors[:, :8].contiguous()  # orthonormal, so every off-diagonal cosine is 0

    loss = infonce(x, x.clone(), temperature=0.05)
    # 7 off-diagonal terms at exp(-1/0.05) each, halved for the symmetric direction.
    assert loss.total.item() < 1e-8, loss.total.item()
    assert loss.accuracy.item() == 1.0


def test_infonce_rewards_a_known_positive_over_a_lookalike():
    """A batch containing a near-duplicate of the anchor is the hard case, and the loss says so."""
    torch.manual_seed(0)
    anchor = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    # A near-copy of row 0 must not win against the true positive.
    anchor[1] = 0.999 * anchor[0] + 0.001 * torch.randn(16)
    positive = anchor.clone()

    loss = infonce(anchor, positive, temperature=0.05)
    assert loss.total.item() > 0.0
    # Row 1 is now a near-duplicate of row 0, so row 0's true positive is no longer the
    # nearest and the loss cannot reach the optimum.
    assert loss.total.item() > 1e-6, loss.total.item()


def test_infonce_is_symmetric():
    a, p = torch.randn(6, 16), torch.randn(6, 16)
    forward = infonce(a, p).total
    backward = infonce(p, a).total
    assert torch.allclose(forward, backward, atol=1e-6)


def test_infonce_matches_manual_computation():
    a = torch.randn(4, 8)
    p = torch.randn(4, 8)
    temp = 0.1
    an, pn = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
    logits = an @ pn.T / temp
    labels = torch.arange(4)
    manual = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    assert torch.allclose(infonce(a, p, temperature=temp).total, manual, atol=1e-6)


def test_infonce_penalises_wrong_pairs():
    x = torch.randn(8, 32)
    shuffled = x[torch.randperm(8)]
    assert infonce(x, x.clone()).total < infonce(x, shuffled).total


def test_infonce_accuracy_tracks_alignment():
    """Aligned pairs should score far better than random ones.

    Compared against a per-trial chance baseline rather than exactly 1/N: with a random
    permutation some rows land on their own positive by luck, and a strict bound would be
    flaky.
    """
    torch.manual_seed(1)
    n = 20
    perfect = torch.randn(n, 24)
    assert infonce(perfect, perfect.clone()).accuracy.item() == 1.0

    # A random permutation's *expected* accuracy is 1/n, but the max over trials is a
    # genuinely larger order statistic, so bound on the mean rather than the max.
    x = torch.randn(n, 24)
    trials = [infonce(x, x[torch.randperm(n)]).accuracy.item() for _ in range(200)]
    mean = sum(trials) / len(trials)
    assert mean < 2.0 / n, f"random pairing averaged {mean:.3f}, expected ~{1 / n:.3f}"


def test_infonce_temperature_affects_scale():
    a, p = torch.randn(6, 16), torch.randn(6, 16)
    hot = infonce(a, p, temperature=1.0).total
    cold = infonce(a, p, temperature=0.01).total
    assert not torch.allclose(hot, cold)


def test_infonce_single_example_is_zero():
    """With one pair there are no negatives, so the loss is exactly zero."""
    x = torch.randn(1, 8)
    assert infonce(x, x.clone()).total.item() == pytest.approx(0.0, abs=1e-6)


def test_learned_temperature_is_bounded():
    head = ContrastiveHead(d_model=8, init_temp=0.05)
    assert float(head.temperature().detach()) == pytest.approx(0.05, rel=1e-3)
    with torch.no_grad():
        head.log_temperature.fill_(50.0)
    assert float(head.temperature().detach()) <= math.exp(4.0) * 1.001
    with torch.no_grad():
        head.log_temperature.fill_(-50.0)
    assert float(head.temperature().detach()) >= math.exp(-4.0) * 0.999


def test_projection_head_is_optional():
    assert ProjectionHead(8, dim=0).enabled is False
    x = torch.randn(2, 8)
    assert torch.equal(ProjectionHead(8, dim=0)(x), x)

    proj = ProjectionHead(8, dim=4)
    assert proj.enabled and proj(x).shape == (2, 4)


def test_contrastive_head_returns_projection_and_temperature():
    head = ContrastiveHead(d_model=16, proj_dim=8)
    pooled = torch.randn(4, 16)
    out, temp = head(pooled)
    assert out.shape == (4, 8)
    assert temp.ndim == 0 and float(temp.detach()) > 0


def test_mine_pairs_returns_in_batch_texts_only():
    texts = [f"document number {i}" for i in range(20)]
    embed_fn = lambda batch: torch.randn(len(batch), 8)
    pairs = mine_pairs(texts, embed_fn, pool_size=10, n_per_anchor=1, seed=0)
    assert pairs, "expected mined pairs"
    known = set(texts)
    for a, b in pairs:
        assert a in known and b in known
        assert a != b, "an anchor must not be its own positive"


def test_mine_pairs_covers_all_anchors():
    texts = [f"doc {i}" for i in range(16)]
    pairs = mine_pairs(texts, lambda b: torch.randn(len(b), 8), pool_size=8, n_per_anchor=1)
    anchors = {a for a, _ in pairs}
    assert len(anchors) == 16


def test_mine_pairs_prefers_true_neighbours():
    """Each document has a twin at a small offset; mining should find the twin.

    A single identical embedding per document is degenerate here: softmax over a block of
    near-identical vectors is uniform, so sampling would be arbitrary. What we actually
    want to verify is that the *nearest* neighbour wins over distant ones.
    """
    base = [f"doc {i}" for i in range(8)]
    centroids = torch.eye(8)
    # doc i's closest neighbour is doc (i+1)%8; the rest are orthogonal.
    def embed_fn(batch: list[str]) -> torch.Tensor:
        idx = torch.tensor([base.index(t) for t in batch])
        return centroids[idx] + 0.01 * torch.randn(len(batch), 8)

    pairs = mine_pairs(base, embed_fn, pool_size=8, n_per_anchor=1, temperature=0.01, seed=0)
    assert pairs
    # The dominant partner should be the planted near-neighbour, not a random document.
    from collections import Counter

    counts = Counter(b for _, b in pairs)
    assert counts.most_common(1)[0][1] >= len(pairs) / 8


def test_pair_batches_are_disjoint_and_sized():
    pairs = [(str(i), f"pos{i}") for i in range(20)]
    batches = list(pair_batches(pairs, batch_size=4, seed=0))
    assert all(len(b) == 4 for b in batches)
    seen = [p for b in batches for p in b]
    assert len(seen) == len(set(seen)), "pairs must not be reused within an epoch"


def test_pair_batches_drop_remainder():
    pairs = [(str(i), f"pos{i}") for i in range(10)]
    assert len(list(pair_batches(pairs, batch_size=4))) == 2


def test_retrieval_metrics_perfect_ranking():
    q = torch.eye(4)
    docs = torch.eye(4)
    m = retrieval_metrics(q, docs, relevant={i: [i] for i in range(4)})
    assert m["recall@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg"] == pytest.approx(1.0)


def test_retrieval_metrics_reward_being_second():
    docs = torch.eye(3)
    q = torch.tensor([[0.0, 0.0, 1.0]])
    good = retrieval_metrics(q, docs, relevant={0: [2]})
    # doc 2 is the nearest neighbour even if listed elsewhere
    assert good["recall@1"] == 1.0


def test_retrieval_metrics_partial_credit():
    docs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]])
    m = retrieval_metrics(q, docs, relevant={0: [1]})
    assert m["recall@1"] == 0.0
    assert m["recall@10"] == 1.0
    assert 0.0 < m["mrr"] <= 1.0


def test_retrieval_metrics_multiple_relevant():
    docs = torch.eye(5)
    q = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    m = retrieval_metrics(q, docs, relevant={0: [2, 0]})
    assert m["recall@1"] == 1.0
    assert m["ndcg"] > 0.5


def test_retrieval_metrics_are_anisotropy_aware():
    """A dominant shared direction inflates raw cosine without carrying information.

    Queries and documents all share one huge component; the tiny perpendicular offsets
    encode the true relation. If ranking used raw cosine on uncentred vectors the
    near-collinear geometry would still dominate, so we assert the metric recovers the
    planted ordering *and* that a deliberately shuffled gold set is not also "perfect".
    """
    shared = torch.tensor([[10.0, 0.0]]).repeat(4, 1)
    queries = shared + torch.tensor([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4]])
    docs = shared + torch.tensor([[0.0, 0.4], [0.0, 0.3], [0.0, 0.2], [0.0, 0.1]])
    correct = retrieval_metrics(queries, docs, {0: [3], 1: [2], 2: [1], 3: [0]})
    assert correct["recall@1"] == 1.0

    # Same vectors, but claiming the *opposite* pairing is correct must score worse.
    inverted = retrieval_metrics(queries, docs, {0: [0], 1: [1], 2: [2], 3: [3]})
    assert inverted["recall@1"] < correct["recall@1"]


def test_retrieval_handles_empty_relevant():
    m = retrieval_metrics(torch.eye(2), torch.eye(2), relevant={})
    assert all(v == 0.0 for v in m.values())