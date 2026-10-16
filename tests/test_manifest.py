"""Tests for the manifest module (Ticket #13).

The manifest is a byteified artifact's only entry point.  Every field is
validated, and the round-trip through JSON must preserve every value.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from byteify import GateResults, Manifest, from_checkpoint, resolve_token_bytes


@pytest.fixture()
def manifest():
    """A minimal but valid manifest."""
    return Manifest(
        schema_version="1",
        architecture="ettin-encoder-17m",
        budget=64,
        max_bytes=512,
        source_id="jhu-clsp/ettin-encoder-17m",
        source_revision="abc123",
        family="wordpiece",
        byte_stream_rule="raw-utf8",
        vocab_size=30522,
        n_byte_values=260,
        gate=GateResults(
            cos_emb=0.92,
            sim_corr=0.89,
            boundary_acc=0.94,
            nn_agree=0.82,
            gate_passed=True,
        ),
        seed=42,
        versions={"python": "3.12"},
        encoder_arch={
            "layers": 1,
            "backend": "attention",
            "window": 128,
            "ffn_hidden": 1024,
            "boundary_predictor": "mlp",
        },
        source_d_model=768,
        source_n_layers=12,
        exported_at="2026-01-01T00:00:00",
    )


def _round_trip(manifest: Manifest) -> Manifest:
    """Serialize to JSON and deserialize back."""
    text = manifest.to_json()
    return Manifest.from_json(text)


# -- round-trip ----------------------------------------------------------


def test_round_trip_preserves_every_field(manifest):
    rebuilt = _round_trip(manifest)
    assert asdict(rebuilt) == asdict(manifest)


def test_round_trip_preserves_gate(manifest):
    rebuilt = _round_trip(manifest)
    assert rebuilt.gate is not None
    assert rebuilt.gate.cos_emb == manifest.gate.cos_emb
    assert rebuilt.gate.sim_corr == manifest.gate.sim_corr
    assert rebuilt.gate.gate_passed == manifest.gate.gate_passed


def test_round_trip_with_no_gate(manifest):
    no_gate_manifest = Manifest(**{
        k: v for k, v in asdict(manifest).items() if k != "gate"
    })
    rebuilt = _round_trip(no_gate_manifest)
    assert rebuilt.gate is None


# -- file I/O -----------------------------------------------------------


def test_write_to_and_read_back(manifest, tmp_path: Path):
    out = tmp_path / "manifest.json"
    manifest.write_to(out)
    assert out.exists()
    loaded = Manifest.from_file(out)
    assert asdict(loaded) == asdict(manifest)


def test_write_to_creates_parent(manifest, tmp_path: Path):
    out = tmp_path / "sub" / "manifest.json"
    manifest.write_to(out)
    assert out.exists()


# -- validation ---------------------------------------------------------


def test_validate_passes_for_valid_manifest(manifest):
    assert manifest.validate() == []


def test_validate_rejects_empty_source_id(manifest):
    bad = Manifest(
        **{k: v for k, v in asdict(manifest).items() if k != "source_id"}
    )
    assert "source_id is empty" in bad.validate()


def test_validate_rejects_invalid_family(manifest):
    bad = Manifest(
        **{k: v for k, v in asdict(manifest).items() if k != "family"
       }
    )
    assert "family is empty" in bad.validate()


def test_validate_rejects_invalid_budget(manifest):
    # Explicitly pass budget=0 to trigger the validation error.
    bad = Manifest(
        budget=0,
        **{k: v for k, v in asdict(manifest).items()
           if k not in ("budget", "gate")}
    )
    assert "budget must be > 0" in bad.validate()


def test_validate_rejects_invalid_d_model(manifest):
    bad = Manifest(
        **{k: v for k, v in asdict(manifest).items() if k != "source_d_model"}
    )
    assert "source_d_model must be > 0" in bad.validate()


# -- from_checkpoint ----------------------------------------------------


def _make_ckpt(tmp_path: Path, extra_config: dict | None = None) -> Path:
    """Create a minimal stage1.pt in tmp_path and return its path."""
    pt_path = tmp_path / "stage1.pt"
    ckpt = {
        "encoder": {"dummy": torch.tensor([1.0])},
        "config": {"max_bytes": 100, "seed": 0, **(extra_config or {})},
        "step": 1,
    }
    torch.save(ckpt, pt_path)
    return pt_path


def _load_real_tokenizer():
    """Load bert-base-uncased for from_checkpoint tests."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    class _Mock:
        strategy = "wordpiece"
        def get_vocab(self):
            return tok.get_vocab()
    return _Mock()


def test_from_checkpoint_builds_manifest(tmp_path: Path):
    """from_checkpoint reads a .pt file and produces a Manifest."""
    _make_ckpt(tmp_path, {
        "model_id": "test-model",
        "local_layers": 2,
        "local_backend": "attention",
        "local_window": 64,
        "local_ffn_hidden": 512,
        "boundary_predictor": "mlp",
        "n_patches": 32,
        "n_byte_values": 260,
        "seed": 7,
    })

    class MockSource:
        d_model = 768
        n_layers = 12

    m = from_checkpoint(
        tmp_path, "test-model", source=MockSource(),
        tokenizer=_load_real_tokenizer(),
        family="wordpiece",
    )
    assert m.source_id == "test-model"
    assert m.source_d_model == 768
    assert m.source_n_layers == 12
    assert m.family == "wordpiece"
    assert m.byte_stream_rule == "raw-utf8"
    assert m.encoder_arch["layers"] == 2
    assert m.seed == 7
    assert m.validate() == []


def test_from_checkpoint_finds_first_pt(tmp_path: Path):
    """If no stage1.pt exists, the first .pt file is used."""
    _make_ckpt(tmp_path)
    (tmp_path / "other.pt").write_text("")  # dummy

    m = from_checkpoint(
        tmp_path, "m", tokenizer=_load_real_tokenizer(),
    )
    assert m.max_bytes == 100


def test_from_checkpoint_no_pt_file(tmp_path: Path):
    """Raises if no .pt file is found."""
    with pytest.raises(FileNotFoundError):
        from_checkpoint(tmp_path, "m")


def test_from_checkpoint_with_history(tmp_path: Path):
    """Gate results are extracted from history.json when present."""
    _make_ckpt(tmp_path)
    history = [
        {"step": 1, "cos_emb_c": 0.92, "sim_corr": 0.89,
         "boundary_acc": 0.95, "nn_agree": 0.80},
    ]
    (tmp_path / "history.json").write_text(json.dumps(history))

    m = from_checkpoint(tmp_path, "m", tokenizer=_load_real_tokenizer())
    assert m.gate is not None
    assert m.gate.cos_emb == 0.92
    assert m.gate.sim_corr == 0.89
    assert m.gate.gate_passed  # 0.92 >= 0.90 and 0.89 >= 0.85


def test_from_checkpoint_gate_fails_history(tmp_path: Path):
    """Gate is marked failed when fidelity is below thresholds."""
    _make_ckpt(tmp_path)
    history = [
        {"step": 1, "cos_emb_c": 0.80, "sim_corr": 0.70,
         "boundary_acc": 0.90, "nn_agree": 0.50},
    ]
    (tmp_path / "history.json").write_text(json.dumps(history))

    m = from_checkpoint(tmp_path, "m", tokenizer=_load_real_tokenizer())
    assert m.gate is not None
    assert m.gate.cos_emb == 0.80
    assert not m.gate.gate_passed  # 0.80 < 0.90


def test_manifest_family_wordpiece_sets_raw_utf8(tmp_path: Path):
    """WordPiece families get byte_stream_rule='raw-utf8'."""
    _make_ckpt(tmp_path)
    m = from_checkpoint(
        tmp_path, "m", tokenizer=_load_real_tokenizer(),
        family="wordpiece",
    )
    assert m.byte_stream_rule == "raw-utf8"


def test_manifest_validation_rejects_invalid_schema():
    bad = Manifest(schema_version="99", source_id="x", family="w",
                   byte_stream_rule="x", source_d_model=1, source_n_layers=1,
                   budget=1, max_bytes=1, vocab_size=1)
    errors = bad.validate()
    assert any("schema version" in e for e in errors)
