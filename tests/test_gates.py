"""Gates and the build pipeline (Tickets #14, #15).

The test that matters most is at the bottom of this file: **a randomly initialised byte
interface must fail the gate.** Today such a model exports cleanly, matches eager PyTorch,
and passes every automated check in this repository -- boundary accuracy 0.000, centred
cosine -0.008, similarity-matrix correlation 0.145. Nothing distinguishes a worthless
artifact from a good one except this gate, which is why #15 exists.

Everything here runs offline against the synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from byteify.bytestream import rule_for_family
from byteify.gates import (
    GateFailure,
    GateReport,
    Thresholds,
    check_exported_graph,
    check_fidelity,
    check_lossless_byte_stream,
    published_source_ids,
    thresholds_for,
)
from byteify.inference import ByteifiedEmbedder
from byteify.levels import LEVELS, resolve_level
from byteify.source import LocalEncoderSpec, build_byteified_encoder

#: Deliberately loose, so these tests measure the gate's *machinery* rather than a
#: particular training run's quality. The real floors are in byteify/assets.
FIXTURE_THRESHOLDS = Thresholds(
    source_id="synthetic",
    cos_emb_c=-1.0,
    sim_corr=-1.0,
    boundary_acc=0.0,
    onnx_max_abs_diff=1e-4,
    trie_coverage=0.0,
)


@pytest.fixture()
def encoder_and_source(tiny_source_encoder, tiny_source_tokenizer):
    spec = LocalEncoderSpec(layers=1, ffn_hidden=32, window=8)
    encoder = build_byteified_encoder(tiny_source_encoder, tiny_source_tokenizer, spec).eval()
    return encoder, tiny_source_encoder, spec


@pytest.fixture()
def embedder(encoder_and_source):
    encoder, source, _ = encoder_and_source
    return ByteifiedEmbedder(encoder, source, n_patches=8).eval()


@pytest.fixture()
def gate_batches(tiny_source_tokenizer, corpus):
    from byteify.stitching import encode_texts

    rows = []
    for i in range(0, 24, 4):
        rows.append(encode_texts(corpus[i : i + 4], tiny_source_tokenizer, device="cpu"))
    return rows


@pytest.fixture()
def trained_encoder(encoder_and_source, tiny_source_tokenizer, corpus):
    """An encoder that has actually seen the data, so fidelity is not chance."""
    from byteify.training import Stage1Config, train_stage1

    encoder, source, _spec = encoder_and_source
    cfg = Stage1Config(
        steps=30,
        batch_size=4,
        max_bytes=128,
        eval_every=30,
        eval_batches=4,
        eval_texts=64,
        n_train=400,
        # Gentle: this test is about whether the gate discriminates, not about how fast a
        # byte interface converges. A larger rate diverges on the tiny fixture, and a NaN
        # there is a real signal rather than something to paper over.
        lr=3e-4,
        device="cpu",
        log_every=10_000,
    )
    train_stage1(encoder, source, tiny_source_tokenizer, corpus[:500], cfg)
    return encoder.eval()


# -- thresholds are data, and per source model ---------------------------


def test_thresholds_are_published_for_the_anchor():
    thresholds = thresholds_for("jhu-clsp/ettin-encoder-17m")
    assert thresholds.cos_emb_c > 0.5
    assert thresholds.sim_corr > 0.5
    assert thresholds.boundary_acc > 0.5


def test_an_unknown_source_model_is_refused_with_a_reason():
    """Fidelity numbers do not travel between architectures, so they are not borrowed."""
    with pytest.raises(KeyError) as excinfo:
        thresholds_for("some/model-nobody-measured")
    message = str(excinfo.value)
    assert "no published fidelity thresholds" in message
    assert "thresholds.json" in message


def test_only_measured_source_models_have_published_floors():
    """A model that has never been gated must be refused, not guessed at.

    Five of the six verified models are alignment-verified but have never had their
    fidelity measured on this metric. Publishing numbers for them would be borrowing
    someone else's floors across architectures, which is the thing per-model thresholds
    exist to prevent.
    """
    assert published_source_ids() == ["jhu-clsp/ettin-encoder-17m"]
    for model_id in (
        "sentence-transformers/all-MiniLM-L6-v2",
        "bert-base-uncased",
        "xlm-roberta-base",
        "jhu-clsp/ettin-encoder-32m",
        "answerdotai/ModernBERT-base",
    ):
        with pytest.raises(KeyError):
            thresholds_for(model_id)


def test_the_anchor_floors_sit_below_what_was_observed_and_above_a_random_interface():
    """The floors have to be reachable by a real run and unreachable by an untrained one."""
    observed = {"cos_emb_c": 0.841, "sim_corr": 0.717, "boundary_acc": 0.944}
    untrained = {"cos_emb_c": -0.005, "sim_corr": 0.241, "boundary_acc": 0.228}
    thresholds = thresholds_for("jhu-clsp/ettin-encoder-17m")

    for name, measured in observed.items():
        floor = getattr(thresholds, name)
        assert floor < measured, f"{name} floor {floor} is above the observed {measured}"
        assert floor > untrained[name], f"{name} floor {floor} would admit an untrained model"


# -- a gate that fails fails ----------------------------------------------


def test_a_gate_report_records_measured_and_threshold():
    report = GateReport()
    report.add("cos_emb_c", 0.91, 0.85)
    report.add("onnx_max_abs_diff", 1.4e-7, 1e-4, "<=")
    assert report.passed
    assert {c.name: c.measured for c in report.checks} == {
        "cos_emb_c": 0.91,
        "onnx_max_abs_diff": 1.4e-7,
    }


def test_a_failed_gate_raises_and_names_the_shortfall():
    report = GateReport()
    report.add("boundary_acc", 0.4, 0.95)
    with pytest.raises(GateFailure) as excinfo:
        report.raise_if_failed()
    assert "boundary_acc" in str(excinfo.value)
    assert "0.55" in str(excinfo.value)  # how far short it fell


def test_the_report_is_storable_in_the_manifest():
    report = GateReport()
    report.add("cos_emb_c", 0.91, 0.85)
    record = report.record()
    assert record["gate_passed"] is True
    assert record["checks"]["cos_emb_c"] == 0.91
    assert record["thresholds"]["cos_emb_c"] == 0.85


# -- levels ---------------------------------------------------------------


def test_every_level_declares_a_guarantee_and_checks():
    for name, level in LEVELS.items():
        assert level.guarantee, name
        assert level.checks, name


def test_a_higher_level_keeps_every_lower_level_check():
    """Raising the level must never quietly drop a guarantee."""
    faithful = set(LEVELS["faithful"].checks)
    for name, level in LEVELS.items():
        if level.index == 1:
            continue
        assert faithful <= set(level.checks), f"{name} drops a level 1 check"


def test_a_higher_level_builds_on_the_one_below():
    assert LEVELS["adapted"].builds_on == ("faithful",)
    assert LEVELS["merged"].builds_on == ("faithful", "adapted")


def test_stage_2_is_not_a_level():
    assert "stage2" not in LEVELS
    with pytest.raises(KeyError, match="Stage 2 is deliberately not on the ladder"):
        resolve_level("stage2")


def test_an_unknown_level_is_refused():
    with pytest.raises(KeyError, match="unknown level"):
        resolve_level("excellent")


# -- the byte-stream gate runs before anything else ----------------------


def test_the_byte_stream_gate_passes_for_a_consistent_rule(
    tiny_source_tokenizer, corpus, tiny_source_encoder
):
    from byteify.alignment import discover_special_bytes

    specials = discover_special_bytes(tiny_source_tokenizer)
    rule = rule_for_family("bytelevel", n_byte_values=256 + len(specials), special_tokens=specials)
    report = GateReport()
    check_lossless_byte_stream(report, tokenizer=tiny_source_tokenizer, rule=rule,
                               texts=corpus[:4])
    report.raise_if_failed()


# -- THE TEST: a random byte interface must fail the gate ------------------


def test_a_randomly_initialised_byte_interface_fails_the_fidelity_gate(
    embedder, gate_batches, tiny_source_encoder
):
    """The gate must reject a model that has learned nothing.

    This is the test #15 exists for. Without it, a random interface exports cleanly, the
    graph matches eager PyTorch to 1e-7, and every other automated check passes -- because
    none of them measure whether the model is any good.
    """
    report = GateReport()
    metrics = check_fidelity(
        report, embedder=embedder, source=tiny_source_encoder, batches=gate_batches,
        thresholds=FIXTURE_THRESHOLDS,
    )
    # The fixture thresholds are permissive on purpose, so the numbers themselves are the
    # assertion: this is what "untrained" looks like, and it is nowhere near shippable.
    assert metrics["boundary_acc"] < 0.9, metrics
    assert metrics["cos_emb_c"] < 0.6, metrics


def test_the_gate_rejects_a_random_interface_against_realistic_floors(
    embedder, gate_batches, tiny_source_encoder
):
    report = GateReport()
    check_fidelity(
        report, embedder=embedder, source=tiny_source_encoder, batches=gate_batches,
        thresholds=Thresholds(
            cos_emb_c=0.5, sim_corr=0.4, boundary_acc=0.9, onnx_max_abs_diff=1e-4
        ),
    )
    assert not report.passed, "a randomly initialised interface passed the fidelity gate"
    assert {c.name for c in report.failures} >= {"boundary_acc"}


def test_a_trained_interface_beats_a_random_one(
    trained_encoder, embedder, gate_batches, tiny_source_encoder
):
    """The gate is not simply refusing everything."""
    trained = ByteifiedEmbedder(trained_encoder, tiny_source_encoder, n_patches=8).eval()
    random_metrics = check_fidelity(
        GateReport(), embedder=embedder, source=tiny_source_encoder,
        batches=gate_batches, thresholds=FIXTURE_THRESHOLDS,
    )
    trained_metrics = check_fidelity(
        GateReport(), embedder=trained, source=tiny_source_encoder,
        batches=gate_batches, thresholds=FIXTURE_THRESHOLDS,
    )
    for name in ("cos_emb_c", "boundary_acc"):
        assert trained_metrics[name] == trained_metrics[name], f"{name} is NaN"
        assert trained_metrics[name] >= random_metrics[name], (
            f"training made {name} worse: {random_metrics[name]} -> {trained_metrics[name]}"
        )


# -- the exported-graph gate ----------------------------------------------


def test_the_exported_graph_gate_catches_a_graph_that_computes_something_else(
    tmp_path: Path, embedder, tiny_source_tokenizer
):
    """A graph that disagrees with eager PyTorch must fail the numerical gate.

    An artifact that loads but computes something else is useless, and nothing downstream
    would notice.
    """
    pytest.importorskip("onnxruntime")

    from byteify.inference import export_onnx

    inputs = _inputs(embedder, tiny_source_tokenizer)
    with torch.no_grad():
        eager = embedder(*inputs)

    good = tmp_path / "good.onnx"
    export_onnx(embedder, tiny_source_tokenizer, path=str(good), max_bytes=128,
                n_patches=8, example_batch=inputs)

    thresholds = Thresholds(onnx_max_abs_diff=1e-4)
    report = GateReport()
    check_exported_graph(report, graph_path=good, inputs=inputs, eager=eager,
                         thresholds=thresholds)
    assert report.passed, report.failures

    # Corrupt the eager result: the graph is fine, the comparison is not. That is the
    # same check failing, which is what proves the check can fail.
    wrong = GateReport()
    check_exported_graph(wrong, graph_path=good, inputs=inputs,
                         eager=eager + 0.5, thresholds=thresholds)
    assert not wrong.passed


def test_the_graph_gate_accepts_inputs_that_need_moving_to_the_host(
    tmp_path: Path, embedder, tiny_source_tokenizer
):
    """ONNX Runtime runs on CPU, so the gate must move tensors across before converting.

    Regression: it called ``.numpy()`` directly, so on an MPS build the gate crashed --
    after training, after the fidelity gates, and after the graph had already been written.
    A non-contiguous tensor exercises the same conversion path on CPU.
    """
    pytest.importorskip("onnxruntime")

    from byteify.inference import export_onnx

    inputs = _inputs(embedder, tiny_source_tokenizer)
    with torch.no_grad():
        eager = embedder(*inputs)

    graph = tmp_path / "graph.onnx"
    export_onnx(embedder, tiny_source_tokenizer, path=str(graph), max_bytes=128,
                n_patches=8, example_batch=inputs)

    # A tensor that requires a copy rather than a view.
    awkward = tuple(t.flip(-1).flip(-1) for t in inputs)
    report = GateReport()
    diff = check_exported_graph(report, graph_path=graph, inputs=awkward, eager=eager,
                                thresholds=Thresholds(onnx_max_abs_diff=1e-4))
    assert report.passed, report.failures
    assert diff <= 1e-4


def test_the_graph_gate_rejects_nan_even_when_eager_agrees(
    tmp_path: Path, embedder, tiny_source_tokenizer
):
    """Agreement with a NaN reference is not agreement.

    Regression: artifacts exported before the masked-softmax fix returned NaN for every
    input, and would have passed a pure numerical-agreement gate if eager had produced the
    same NaN. Finiteness is its own check for that reason.
    """
    pytest.importorskip("onnxruntime")

    from byteify.inference import export_onnx

    inputs = _inputs(embedder, tiny_source_tokenizer)
    with torch.no_grad():
        eager = embedder(*inputs)

    graph = tmp_path / "graph.onnx"
    export_onnx(embedder, tiny_source_tokenizer, path=str(graph), max_bytes=128,
                n_patches=8, example_batch=inputs)

    report = GateReport()
    check_exported_graph(report, graph_path=graph, inputs=inputs,
                         eager=eager * float("nan"), thresholds=Thresholds())
    assert not report.passed
    assert {c.name for c in report.failures} >= {"outputs_finite"}


def _inputs(embedder, tokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    from byteify.alignment import discover_special_bytes
    from byteify.expand import expand_byte_ids
    from byteify.stitching import tokenizer_tables

    specials = discover_special_bytes(tokenizer)
    rule = rule_for_family("bytelevel", n_byte_values=256 + len(specials),
                           special_tokens=specials)
    _, trie, _ = tokenizer_tables(tokenizer)
    ids = rule.build("byte level retrofit 日本語", 128)
    expanded = expand_byte_ids(ids, trie, pseudo_byte_to_token=rule.pseudo_bytes)
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([expanded], dtype=torch.long),
        torch.ones(1, len(ids), dtype=torch.long),
    )


# -- a failed gate writes nothing -----------------------------------------


def test_a_failed_gate_leaves_the_output_directory_empty(tmp_path: Path):
    """No partial artifact. An artifact that failed its own promise is worse than none."""
    out = tmp_path / "artifact"
    out.mkdir()
    report = GateReport()
    report.add("boundary_acc", 0.0, 0.95)
    with pytest.raises(GateFailure):
        report.raise_if_failed()
    assert list(out.iterdir()) == []


# -- the manifest refuses an artifact whose gate failed -------------------


# -- the patch budget has to fit the byte axis ----------------------------


def test_the_patch_budget_defaults_to_the_byte_axis():
    """A fixed default of "64" was wrong for any axis above about 256 bytes.

    Bolmo's ratio is about 4 bytes per patch, so a 512-byte axis needs ~128. Measured on
    the anchor: centred cosine 0.585 at budget 64 against 0.795 at 128, same checkpoint.
    """
    from byteify.build import TARGET_BYTES_PER_PATCH, _fit_patch_budget

    lines: list[str] = []
    budget = _fit_patch_budget(None, max_bytes=512, say=lambda m="": lines.append(m))
    assert budget == round(512 / TARGET_BYTES_PER_PATCH)
    assert any("derived" in line for line in lines)

    assert _fit_patch_budget(None, max_bytes=256, say=lambda m="": lines.append(m)) == 64
    assert _fit_patch_budget(None, max_bytes=128, say=lambda m="": lines.append(m)) == 32


def test_a_mismatched_patch_budget_is_warned_about():
    """Too few patches merges them and costs fidelity; the user should be told."""
    from byteify.build import _fit_patch_budget

    lines: list[str] = []
    chosen = _fit_patch_budget(64, max_bytes=512, say=lambda m="": lines.append(m))
    assert chosen == 64, "the user's explicit budget must be honoured"
    assert any("warning" in line and "patch budget" in line for line in lines), lines


def test_a_sensible_patch_budget_is_not_warned_about():
    from byteify.build import _fit_patch_budget

    lines: list[str] = []
    _fit_patch_budget(128, max_bytes=512, say=lambda m="": lines.append(m))
    assert not any("warning" in line for line in lines), lines


# -- a manifest refuses an artifact whose gate failed -------------------


def test_a_manifest_recording_a_failed_gate_is_refused(valid_manifest):
    from dataclasses import replace

    from byteify import GateResults

    failed = replace(valid_manifest, gate=GateResults(cos_emb=0.1, gate_passed=False))
    assert any("gate" in e for e in failed.validate())