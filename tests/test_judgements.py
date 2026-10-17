"""Judgements: the labels a level-2 build is graded against.

Level 2's added guarantee is retrieval quality on the corpus a practitioner supplied, and
judgements are how that claim becomes checkable. Two things about them are load-bearing
and neither is obvious from the format:

**The candidate pool belongs to its query.** Ranking a query against every document in a
cora makes chance depend on how large the corpus is, so two artifacts fitted to different
corpora produce numbers that cannot be compared. Ranking a query against the candidates
that were supplied with it is a decision a practitioner can act on, and its chance level
is ``1/k`` for that query -- which is reported rather than assumed.

**A label nobody supplied must not be invented.** A row whose positive is missing from its
own pool, or whose target ties, is dropped. Repairing it -- treating the first candidate as
the answer, or searching for a near match -- would put a number on a label that does not
exist, and that number would then gate a build.
"""

from __future__ import annotations

import json

import pytest

from byteify.training import Judgements, load_judgements


def test_a_pool_row_resolves_its_positive_by_index(tmp_path):
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        json.dumps(
            {"query": "q0", "candidates": ["a", "b", "c"], "positive": 2}
        )
        + "\n"
        + json.dumps(
            {"query": "q1", "candidates": ["d", "e"], "positive": "e"}
        )
        + "\n",
        encoding="utf-8",
    )

    judgements = load_judgements(str(path))

    assert judgements.queries == ["q0", "q1"]
    assert judgements.candidates == [["a", "b", "c"], ["d", "e"]]
    # Both spellings of "which one" resolve to the same thing: an index into the pool.
    assert judgements.relevant == [[2], [1]]
    assert judgements.chance_recall_at_1 == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_the_two_field_format_still_loads_as_a_pool_of_one(tmp_path):
    """`{"query": q, "positive": doc}` predates candidate pools and has to keep working."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({"query": "q", "positive": "the document"}) + "\n", encoding="utf-8"
    )

    judgements = load_judgements(str(path))

    assert judgements.candidates == [["the document"]]
    assert judgements.relevant == [[0]]
    # A pool of one is trivially perfectly ranked, which is exactly why the pool form is
    # the one to write: this number certifies nothing and the chance level says so.
    assert judgements.chance_recall_at_1 == 1.0


def test_a_json_query_map_loads_the_same_way(tmp_path):
    path = tmp_path / "judgements.json"
    path.write_text(
        json.dumps({"q0": {"positive": "a"}, "q1": "b"}), encoding="utf-8"
    )

    judgements = load_judgements(str(path))

    assert judgements.queries == ["q0", "q1"]
    assert judgements.relevant == [[0], [0]]


def test_a_positive_outside_its_own_pool_is_dropped_rather_than_repaired(tmp_path):
    """Inventing a label would gate a build on a judgement nobody made."""
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        json.dumps({"query": "q", "candidates": ["a", "b"], "positive": "somewhere else"})
        + "\n"
        + json.dumps({"query": "q2", "candidates": ["a", "b"], "positive": 9})
        + "\n"
        + json.dumps({"query": "q3", "candidates": ["a", "b"], "positive": "b"})
        + "\n",
        encoding="utf-8",
    )

    judgements = load_judgements(str(path))

    assert judgements.queries == ["q3"]


def test_a_file_with_nothing_usable_in_it_is_an_error_not_an_empty_gate(tmp_path):
    """Silently grading zero decisions would report a perfect score on nothing."""
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_judgements(str(path))
    assert "no usable judgements" in str(excinfo.value)


def test_no_judgements_at_all_is_none_not_an_error():
    """Optional means absent is normal; only an unusable file is a problem."""
    assert load_judgements(None) is None


def test_retrieval_is_graded_within_each_querys_own_pool():
    """The metric's chance level has to come from the pools, not from a global corpus."""
    import torch

    from byteify.contrastive import retrieval_over_judgements

    judgements = Judgements(
        queries=["q0", "q1"],
        candidates=[["a", "b"], ["c", "d", "e"]],
        relevant=[[0], [2]],
    )
    vectors = {"q0": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0],
               "q1": [0.0, 1.0], "c": [1.0, 0.0], "d": [0.0, 1.0], "e": [-1.0, 0.0]}

    def embed(texts):
        return torch.tensor([vectors[t] for t in texts])

    metrics = retrieval_over_judgements(embed, judgements)

    assert metrics["recall@1"] == 0.5        # q0 ranks its gold first, q1 does not
    assert metrics["chance_recall_at_1"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert metrics["n_decisions"] == 2


def test_each_distinct_candidate_text_is_embedded_once():
    """Embedding is a byte-level forward pass; re-embedding a shared candidate is waste."""
    import torch

    from byteify.contrastive import retrieval_over_judgements

    judgements = Judgements(
        queries=["q0", "q1", "q2"],
        candidates=[["shared", "x"], ["shared", "y"], ["shared", "z"]],
        relevant=[[0], [0], [0]],
    )
    seen: list[str] = []

    def embed(texts):
        seen.extend(texts)
        return torch.eye(len(texts), 4)[torch.arange(len(texts)) % 4]

    retrieval_over_judgements(embed, judgements)

    # Three queries plus four distinct candidate texts, each embedded exactly once.
    assert seen.count("shared") == 1
    assert sorted(seen) == ["q0", "q1", "q2", "shared", "x", "y", "z"][:7]
    assert len(seen) == 7