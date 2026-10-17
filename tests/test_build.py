"""``byteify build`` end to end, offline, in seconds (Tickets #14, #15).

The highest seam available: produce an artifact, load it only through its manifest, encode
text. It covers the byte-stream rule, alignment, trie construction, the recipe, the gates,
the export and the manifest round-trip, because the product *is* the artifact.

Everything runs on the synthetic fixture. Two bugs in this path -- an ONNX example batch
built on the wrong device, and a `.to()` called on a tuple -- were only reachable by running
a ten-minute training job first. That is the cost of not having this file, and these tests
take a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from byteify.gates import GateFailure
from byteify.inference import ByteifiedEmbedder


@pytest.fixture()
def synthetic_source(tmp_path: Path, tiny_tokenizer):
    """A source model on disk whose id is a key in the published thresholds.

    Level 1 refuses a source model with no published fidelity floors, so an end-to-end test
    of the *gate* path needs one. The floors are temporarily set to zero: what is under test
    here is the plumbing -- export, write, load, encode -- not whether 8 training steps are
    enough fidelity.
    """
    from byteify import gates
    from tests.synthetic import build_source_model

    path = tmp_path / "source"
    build_source_model("modernbert", tiny_tokenizer).save_pretrained(path)
    # A local directory passed as --model must hold the tokenizer too, exactly as a
    # published repository does. `byteify build` resolves both from that one path.
    tiny_tokenizer.save_pretrained(path)

    original = gates._published
    gates._published = lambda: {
        str(path): gates.Thresholds(
            source_id=str(path),
            cos_emb_c=-1.0,
            sim_corr=-1.0,
            boundary_acc=0.0,
            onnx_max_abs_diff=1e-4,
            trie_coverage=0.0,
        )
    }
    gates.thresholds_for.__wrapped__ if hasattr(gates.thresholds_for, "__wrapped__") else None
    try:
        yield str(path)
    finally:
        gates._published = original


def _corpus_file(tmp_path: Path) -> str:
    """A reference corpus as JSONL, one document per line.

    Two things about the text matter. ``.txt`` is read as *documents*, not lines --
    ``iter_texts`` accumulates until 4000 characters -- so a paragraph-per-line file yields
    a handful of long documents rather than the many short ones a test wants. And the text
    is drawn from the corpus the fixture tokenizer was trained on: an ``[UNK]`` has no byte
    string, so ``align_text`` drops it and the training stream is shorter than the text,
    which the byte-stream gate correctly refuses. That is a property of a 400-token
    vocabulary, not of the build.
    """
    from tests.synthetic import synthetic_corpus

    path = tmp_path / "corpus.jsonl"
    # Only sentences clearing `iter_texts`' 64-character floor are usable, and roughly a
    # third of the fixture's sentences are shorter than that, so oversample.
    with path.open("w", encoding="utf-8") as handle:
        for text in synthetic_corpus(900):
            handle.write(json.dumps({"text": text}) + "\n")
    return str(path)


def _build(synthetic_source: str, out: Path, tmp_path: Path, **overrides):
    from byteify.build import build_artifact

    kwargs = {
        "model": synthetic_source,
        "level": "faithful",
        "data": [_corpus_file(tmp_path)],
        "output_dir": out,
        "steps": 4,
        "batch_size": 2,
        "max_bytes": 96,
        "patch_budget": 24,
        "local_layers": 1,
        "local_ffn": 32,
        "local_window": 16,
        "device": "cpu",
        "quiet": True,
    }
    kwargs.update(overrides)
    return build_artifact(**kwargs)


def test_build_writes_a_complete_loadable_artifact(synthetic_source, tmp_path: Path):
    """The whole vertical slice: train, gate, export, write, load, encode."""
    out = tmp_path / "artifact"
    manifest = _build(synthetic_source, out, tmp_path)

    for name in ("manifest.json", "artifact.onnx", "trie.json", "encoder.pt"):
        assert (out / name).exists(), f"{name} missing from the artifact"

    assert manifest.validate() == []
    assert manifest.gate.gate_passed
    assert manifest.source_id == synthetic_source

    # Loads through the manifest alone, and embeds.
    reloaded_manifest, embedder = __import__(
        "byteify.manifest", fromlist=["load_artifact"]
    ).load_artifact(out)
    assert reloaded_manifest.budget == manifest.budget
    embeddings = embedder.encode_text(["a short document", "another one entirely"])
    assert embeddings.shape[0] == 2
    assert torch.isfinite(embeddings).all()


def test_the_manifest_records_the_level_and_its_measured_guarantee(
    synthetic_source, tmp_path: Path
):
    out = tmp_path / "artifact"
    manifest = _build(synthetic_source, out, tmp_path)

    assert manifest.gate.level == "faithful"
    assert manifest.gate.guarantee
    declared = manifest.gate.declared
    for name in ("cos_emb_c", "sim_corr", "boundary_acc"):
        assert declared[name] is not None, name
        assert name in manifest.gate.thresholds, name


def test_the_exported_graph_is_recorded_as_agreeing_with_eager(
    synthetic_source, tmp_path: Path
):
    out = tmp_path / "artifact"
    manifest = _build(synthetic_source, out, tmp_path)

    assert manifest.gate.checks["outputs_finite"] == 1.0
    assert manifest.gate.checks["onnx_max_abs_diff"] <= 1e-4


def test_a_failed_gate_writes_nothing(synthetic_source, tmp_path: Path):
    """The load-bearing property. An artifact that failed its promise is worse than none."""
    from byteify import gates

    out = tmp_path / "artifact"
    original = gates._published
    gates._published = lambda: {
        synthetic_source: gates.Thresholds(
            source_id=synthetic_source,
            cos_emb_c=0.999,      # unreachable
            sim_corr=0.999,
            boundary_acc=0.999,
            onnx_max_abs_diff=1e-4,
            trie_coverage=0.0,
        )
    }
    try:
        with pytest.raises(GateFailure):
            _build(synthetic_source, out, tmp_path)
    finally:
        gates._published = original

    assert not out.exists() or list(out.iterdir()) == []


def test_an_unknown_level_is_refused_before_anything_happens(synthetic_source, tmp_path):
    out = tmp_path / "artifact"
    with pytest.raises(KeyError, match="Stage 2 is deliberately not on the ladder"):
        _build(synthetic_source, out, tmp_path, level="stage2")
    assert not out.exists()


def test_an_unpublished_source_model_is_refused(tmp_path: Path):
    """No floors means no floors -- refused before training, not guessed at."""
    from byteify.build import build_artifact

    with pytest.raises(KeyError, match="no published fidelity thresholds"):
        build_artifact(
            model="some/model-nobody-measured",
            level="faithful",
            data=[_corpus_file(tmp_path)],
            output_dir=tmp_path / "artifact",
            steps=1,
            quiet=True,
        )


def test_too_little_reference_data_is_refused_with_the_minimum(
    synthetic_source, tmp_path: Path
):
    from byteify.build import build_artifact

    tiny = tmp_path / "tiny.txt"
    tiny.write_text("one document only", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 64"):
        build_artifact(
            model=synthetic_source,
            level="faithful",
            data=[str(tiny)],
            output_dir=tmp_path / "artifact",
            steps=1,
            quiet=True,
        )


def test_an_existing_artifact_is_not_silently_overwritten(synthetic_source, tmp_path: Path):
    out = tmp_path / "artifact"
    _build(synthetic_source, out, tmp_path)
    before = (out / "manifest.json").read_text()

    with pytest.raises(FileExistsError, match="does not overwrite"):
        _build(synthetic_source, out, tmp_path)
    assert (out / "manifest.json").read_text() == before


def test_a_dry_run_writes_nothing(synthetic_source, tmp_path: Path):
    out = tmp_path / "artifact"
    assert _build(synthetic_source, out, tmp_path, dry_run=True) is None
    assert not out.exists()


def test_encode_reads_the_artifact_and_refuses_over_long_text(
    synthetic_source, tmp_path: Path
):
    """The documented serving path, and its loud failure on input it cannot hold."""
    from byteify.cli import _onnx_inputs

    out = tmp_path / "artifact"
    manifest = _build(synthetic_source, out, tmp_path)

    class _Args:
        max_bytes = None

    rows = _onnx_inputs(manifest, out, "a short document", _Args())
    assert rows["byte_mask"].sum() > 0
    assert rows["byte_ids"].shape == rows["expanded_ids"].shape == rows["byte_mask"].shape

    too_long = "word " * (manifest.max_bytes)
    with pytest.raises(ValueError, match="cannot take longer input"):
        _onnx_inputs(manifest, out, too_long, _Args())


def test_encode_produces_finite_embeddings_from_the_artifact(
    synthetic_source, tmp_path: Path, capsys
):
    from byteify.cli import main

    out = tmp_path / "artifact"
    _build(synthetic_source, out, tmp_path)

    assert main(["encode", str(out), "--text", "a short document"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    vector = payload["embedding"]
    # One value per model dimension; the synthetic source is 32 wide.
    assert len(vector) == 32
    assert all(v == v for v in vector), "a NaN reached the serving path"  # noqa: PLR0124
    assert payload["bytes"] > 0


def test_encode_reports_a_bad_artifact_as_a_message_not_a_traceback(tmp_path, capsys):
    from byteify.cli import main

    assert main(["encode", str(tmp_path), "--text", "hello"]) == 1
    assert "error:" in capsys.readouterr().out


def test_the_byte_axis_is_exactly_what_the_manifest_declares(synthetic_source, tmp_path):
    out = tmp_path / "artifact"
    _build(synthetic_source, out, tmp_path, max_bytes=96, patch_budget=24)

    embedder = ByteifiedEmbedder.load(out)
    assert embedder.byte_axis == 96
    assert embedder.n_patches == 24
    assert embedder.byte_stream_rule.rule == "nfc"