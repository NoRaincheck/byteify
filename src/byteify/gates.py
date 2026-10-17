"""Gates: a level's guarantee, expressed as numbers that hard-fail (ADR 0010).

A gate is not a warning. If a check fails, the command stops and no artifact is written,
because an artifact that failed its own promise is worse than no artifact: it loads, it
runs, and it is wrong.

**Thresholds are per source model, published as data.** Fidelity numbers do not transfer
between architectures -- a 7-layer 256-wide encoder and a 22-layer 768-wide one have no
reason to land on the same centred cosine. Shipping level 1 for a source model therefore
includes establishing its thresholds, and pointing byteify at a model with no published
thresholds is refused rather than measured against someone else's numbers.

The thresholds in ``thresholds.json`` are measured floors for the models listed there. A
source model with no entry cannot be byteified at a level yet, and the tool says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

#: Where the thresholds live, relative to the package. Editable, and the reason they are
#: data rather than constants.
THRESHOLD_RESOURCE = "thresholds.json"


class GateFailure(RuntimeError):
    """A gate failed. The command must stop and write nothing."""

    def __init__(self, report: GateReport) -> None:
        self.report = report
        failed = "; ".join(
            f"{c.name} {c.measured:.4g} {c.comparison} {c.threshold:.4g} (short by "
            f"{abs(c.measured - c.threshold):.4g})"
            for c in report.checks
            if not c.passed
        )
        super().__init__(
            f"{len(report.failures)} of {len(report.checks)} gate checks failed: {failed}"
        )


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The floors a source model's artifact must clear to be worth shipping."""

    source_id: str = ""
    architecture: str = ""
    #: Centred cosine of pooled sentence embeddings against the source model.
    cos_emb_c: float = 0.0
    #: Correlation of the two pairwise similarity matrices.
    sim_corr: float = 0.0
    #: Agreement on boundary decisions.
    boundary_acc: float = 0.0
    #: Max absolute difference between the exported graph and eager PyTorch.
    onnx_max_abs_diff: float = 1e-4
    #: Suffix-trie coverage floor, measured over the reference data.
    trie_coverage: float = 0.0
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Thresholds:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@lru_cache(maxsize=1)
def _published() -> dict[str, Thresholds]:
    text = (
        resources.files("byteify")
        .joinpath("assets", THRESHOLD_RESOURCE)
        .read_text("utf-8")
    )
    return {k: Thresholds.from_dict(v) for k, v in json.loads(text).items()}


def published_source_ids() -> list[str]:
    return sorted(_published())


def thresholds_for(source_id: str) -> Thresholds:
    """The thresholds for a source model, or a refusal naming what is supported."""
    known = _published()
    if source_id in known:
        return known[source_id]
    raise KeyError(
        f"no published fidelity thresholds for {source_id!r}.\n"
        f"Fidelity numbers do not transfer between architectures, so byteify will not "
        f"measure this model against another one's floors.\n"
        f"Measure it first (see RESULTS.md) and add an entry to "
        f"byteify/assets/{THRESHOLD_RESOURCE}."
    )


@dataclass(frozen=True, slots=True)
class Check:
    """One measured number against one floor."""

    name: str
    measured: float
    threshold: float
    comparison: str = ">="

    @property
    def passed(self) -> bool:
        if self.comparison == ">=":
            return self.measured >= self.threshold
        if self.comparison == "<=":
            return self.measured <= self.threshold
        raise ValueError(f"unknown comparison {self.comparison!r}")


@dataclass
class GateReport:
    """The result of running a level's checks."""

    checks: list[Check] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, measured: float, threshold: float, comparison: str = ">=") -> Check:
        check = Check(name=name, measured=float(measured), threshold=float(threshold),
                      comparison=comparison)
        self.checks.append(check)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def record(self) -> dict[str, Any]:
        """The report as it is stored in the manifest."""
        return {
            "checks": {c.name: c.measured for c in self.checks},
            "thresholds": {c.name: c.threshold for c in self.checks},
            "gate_passed": self.passed,
            **self.extra,
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise GateFailure(self)


def check_lossless_byte_stream(
    report: GateReport,
    *,
    tokenizer,
    rule,
    texts: list[str],
) -> None:
    """Every reference text must produce the identical bytes at training and inference.

    This is checked before anything else because it is the one failure that cannot be
    caught downstream: an artifact built on a corrupted byte stream still satisfies its
    fidelity guarantee, because that guarantee is measured against the corrupted stream it
    was built from.
    """
    from .alignment import align_text, discover_special_bytes

    specials = discover_special_bytes(tokenizer)
    mismatches = []
    for text in texts:
        trained = list(align_text(text, tokenizer, special_token_ids=specials).byte_ids)
        inferred = rule.build(text, 10_000)
        if inferred != trained:
            mismatches.append(text[:40])
    report.add("byte_stream_lossless", float(len(mismatches)), 0.0, "<=")
    if mismatches:
        report.extra["byte_stream_mismatches"] = mismatches


def check_fidelity(
    report: GateReport,
    *,
    embedder,
    source,
    batches,
    thresholds: Thresholds,
) -> dict[str, float]:
    """Boundary agreement and similarity structure, centred."""
    from .fidelity import _budgeted_patch_reps, measure_fidelity

    metrics = measure_fidelity(
        lambda batch: _budgeted_patch_reps(embedder, batch), source, batches
    )
    report.add("cos_emb_c", metrics["cos_emb_c"], thresholds.cos_emb_c)
    report.add("sim_corr", metrics["sim_corr"], thresholds.sim_corr)
    report.add("boundary_acc", metrics["boundary_acc"], thresholds.boundary_acc)
    report.extra["fidelity"] = metrics
    return metrics


def check_exported_graph(
    report: GateReport,
    *,
    graph_path: Path,
    inputs: tuple[Any, Any, Any],
    eager: Any,
    thresholds: Thresholds,
) -> float:
    """The exported graph must compute what eager PyTorch computes.

    Prior art in this repository and the model for the numerical gate: an artifact that
    loads but computes something else is useless, and nothing downstream would notice.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - depends on the optional extra
        report.extra["onnx_check"] = "skipped: onnxruntime is not installed"
        return float("nan")

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    got = session.run(
        None,
        {
            "byte_ids": inputs[0].numpy(),
            "expanded_ids": inputs[1].numpy(),
            "byte_mask": inputs[2].numpy(),
        },
    )[0]
    diff = float(np.abs(got - eager.detach().cpu().numpy()).max())
    report.add("onnx_max_abs_diff", diff, thresholds.onnx_max_abs_diff, "<=")
    return diff


def check_coverage(report: GateReport, coverage: float | None, thresholds: Thresholds) -> None:
    """How much of real text the suffix trie resolves.

    Not a pass/fail on the model, a statement about where it will be strong: complete for
    byte-level BPE, and much lower for other families.
    """
    if coverage is None:
        return
    report.add("trie_coverage", coverage, thresholds.trie_coverage)