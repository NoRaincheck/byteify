"""Levels: what a practitioner chooses, and what they get for choosing it (ADR 0009).

A level resolves to a recipe plus a **declared** set of checks. The recipe is never part of
the user's contract; the level is the interface. What changes as the level rises is how
well the artifact fits the workload it was fitted for -- **not** fidelity, which is constant
across the ladder because every level starts from a Stage 1 checkpoint.

    level 1  faithful   Stage 1
    level 2  adapted    Stage 1 + contrastive on the supplied corpus
    level 3  merged     + a task-arithmetic delta from a post-trained source variant

Each level declares what it guarantees and gates on it. Failing a gate stops the command
and writes nothing, which is the load-bearing part: it is what makes "different levels" a
ladder of checkable claims rather than a naming convention.

**Stage 2 is not a level** and is not reachable from here. It trains the global model away
from the source model, which is exactly what the fidelity constraint forbids. It remains a
research path; see RESULTS.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Level:
    """One rung: a name, what it runs, and what it promises."""

    name: str
    index: int
    summary: str
    #: What this level guarantees, as a sentence a practitioner can check.
    guarantee: str
    #: The checks it gates on, in the order they run.
    checks: tuple[str, ...]
    #: True when the level needs relevance judgements to reach its full guarantee.
    needs_judgements: bool = False
    #: Stages the level is on top of.
    builds_on: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = [f"level {self.index} `{self.name}` -- {self.summary}", f"  guarantees: {self.guarantee}"]
        lines.append(f"  checks: {', '.join(self.checks)}")
        if self.needs_judgements:
            lines.append(
                "  relevance judgements are optional: without them the retrieval check "
                "is skipped and the rest still gate"
            )
        return "\n".join(lines)


#: The checks every level runs. Present in every level's `checks`, and the reason a higher
#: level never silently loses a lower level's guarantee.
COMMON_CHECKS = ("byte_stream_lossless", "onnx_max_abs_diff", "cos_emb_c", "sim_corr", "boundary_acc")

LEVELS: dict[str, Level] = {
    "faithful": Level(
        name="faithful",
        index=1,
        summary="Stage 1, preserving the source model's behaviour",
        guarantee=(
            "boundary decisions and similarity structure match the source model to the "
            "numbers in byteify/data/thresholds.json"
        ),
        checks=COMMON_CHECKS,
    ),
    "adapted": Level(
        name="adapted",
        index=2,
        summary="Stage 1 then contrastive training on the supplied corpus",
        guarantee=(
            "everything `faithful` guarantees, plus retrieval quality on the supplied "
            "corpus against any supplied judgements"
        ),
        checks=COMMON_CHECKS + ("trie_coverage", "retrieval"),
        needs_judgements=True,
        builds_on=("faithful",),
    ),
    "merged": Level(
        name="merged",
        index=3,
        summary="plus a task-arithmetic delta from a post-trained source variant",
        guarantee=(
            "everything `adapted` guarantees, plus that the delta is present in the "
            "merged weights"
        ),
        checks=COMMON_CHECKS + ("trie_coverage", "retrieval", "delta_present"),
        needs_judgements=True,
        builds_on=("faithful", "adapted"),
    ),
}


def resolve_level(name: str) -> Level:
    """Resolve a level name, refusing one byteify cannot deliver."""
    level = LEVELS.get(name)
    if level is None:
        raise KeyError(
            f"unknown level {name!r}; choose one of {', '.join(sorted(LEVELS))}.\n"
            "Stage 2 is deliberately not on the ladder: it trains the global model away "
            "from the source model, which is what the fidelity guarantee forbids."
        )
    return level


def require_predecessor(level: Level, artifact: dict | None) -> None:
    """Refuse a level whose predecessor artifact is absent or from a different source."""
    if not level.builds_on:
        return
    if artifact is None:
        raise FileNotFoundError(
            f"level `{level.name}` builds on {', '.join(level.builds_on)}; run that level "
            "first so raising the level never costs fidelity silently"
        )


def declare(level: Level, thresholds, metrics: dict, *, extra: dict | None = None) -> dict:
    """The numbers a level's guarantee is stated as, for the manifest."""
    return {
        "level": level.name,
        "level_index": level.index,
        "guarantee": level.guarantee,
        "declared": {
            "cos_emb_c": metrics.get("cos_emb_c"),
            "sim_corr": metrics.get("sim_corr"),
            "boundary_acc": metrics.get("boundary_acc"),
            "cos_emb_raw": metrics.get("cos_emb_raw"),
            "nn_agree": metrics.get("nn_agree"),
            "nn_chance": metrics.get("nn_chance"),
        },
        "thresholds": {
            "cos_emb_c": thresholds.cos_emb_c,
            "sim_corr": thresholds.sim_corr,
            "boundary_acc": thresholds.boundary_acc,
            "onnx_max_abs_diff": thresholds.onnx_max_abs_diff,
            "trie_coverage": thresholds.trie_coverage,
        },
        **(extra or {}),
    }


#: Kept for the type of a gate runner, which callers import rather than pass around.
GateRunner = Callable[..., dict]