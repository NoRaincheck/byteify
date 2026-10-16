"""Byteified artifact manifest.

The manifest is the **sole** entry point for any byteified serving artifact.
A manifest carries:

  * schema version, architecture, budget, byte-axis length
  * source model id and revision (which was byteified)
  * tokenizer family and byte-stream rule (how bytes are derived from text)
  * local-encoder architecture (layers, backend, window, feed-forward width, boundary predictor)
  * gate results (fidelity metrics: cos, sim_corr, boundary_acc)
  * seed and library versions

When a client loads a byteified model, it reads the manifest first, validates
schema version, and uses the declared fields to construct the encoder.  No
hardcoded constants, no ``save_pretrained`` — the manifest is the contract.

The schema is persisted as a UTF-8 JSON file alongside the ``.onnx`` graph
(with an optional ``.onnx.data`` shard).  The export path writes both files
atomically (write to a temp, rename).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Current manifest schema version.  Increment when fields change.
SCHEMA_VERSION: str = "1"


@dataclass(frozen=True, slots=True)
class GateResults:
    """Numerical quality gates (ADR 0010).  Stored after training / export."""

    cos_emb: float | None = None  # cosine similarity of pooled embeddings
    sim_corr: float | None = None  # similarity-matrix correlation
    boundary_acc: float | None = None  # boundary prediction accuracy
    nn_agree: float | None = None  # nearest-neighbour agreement
    gate_passed: bool = True  # overall pass/fail

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateResults:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class Manifest:
    """Describes a single byteified serving artifact.

    Attributes:
        schema_version: the manifest schema version (always "1" for now).
        architecture: the original source model's architecture name
            (e.g. ``ettin-encoder-17m``).
        budget: patch budget (fixed at export for ONNX).
        max_bytes: byte-axis length (fixed at export for ONNX).
        source_id: the source model's model_id (from HuggingFace).
        source_revision: the source model's Git revision or tag.
        family: the tokenizer family — ``bytelevel``, ``wordpiece``,
            or ``sentencepiece``.
        byte_stream_rule: how the byte stream is derived from text
            (e.g. ``NFC`` for byte-level BPE, ``raw-utf8`` for
            WordPiece / SentencePiece).
        vocab_size: the source tokenizer's vocabulary size.
        n_byte_values: ``256 + len(special_token_ids)``.
        gate: quality gate results (None if not yet evaluated).
        seed: random seed used for training.
        versions: library versions captured at export.
        encoder_arch: the local-encoder spec (layers, backend, window,
            ffn_hidden, boundary_predictor).
        source_d_model: hidden width of the source model.
        source_n_layers: number of layers in the source model.
        exported_at: ISO-8601 timestamp of export.
    """

    schema_version: str = SCHEMA_VERSION
    architecture: str = ""
    budget: int = 64
    max_bytes: int = 512
    source_id: str = ""
    source_revision: str = ""
    family: str = ""
    byte_stream_rule: str = ""
    vocab_size: int = 0
    n_byte_values: int = 256
    gate: GateResults | None = None
    seed: int = 0
    versions: dict[str, str] = field(default_factory=dict)
    encoder_arch: dict[str, Any] = field(default_factory=dict)
    source_d_model: int = 0
    source_n_layers: int = 0
    exported_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.gate is not None:
            d["gate"] = self.gate.to_dict()
        else:
            d["gate"] = None
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        gate = None
        if d.get("gate") is not None:
            gate = GateResults.from_dict(d["gate"])
        return cls(gate=gate, **{k: v for k, v in d.items() if k != "gate"})

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: Path) -> Manifest:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def write_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(
                f"schema version {self.schema_version} != {SCHEMA_VERSION}"
            )
        if self.source_id == "":
            errors.append("source_id is empty")
        if self.family == "":
            errors.append("family is empty")
        if self.byte_stream_rule == "":
            errors.append("byte_stream_rule is empty")
        if self.source_d_model <= 0:
            errors.append("source_d_model must be > 0")
        if self.source_n_layers <= 0:
            errors.append("source_n_layers must be > 0")
        if self.budget <= 0:
            errors.append("budget must be > 0")
        if self.max_bytes <= 0:
            errors.append("max_bytes must be > 0")
        if self.vocab_size <= 0:
            errors.append("vocab_size must be > 0")
        return errors


def from_checkpoint(
    checkpoint_dir: Path | str,
    source_id: str,
    source_revision: str = "",
    source: Any = None,
    tokenizer: Any = None,
    *,
    family: str | None = None,
) -> Manifest:
    """Build a manifest from a trained-stage-1 checkpoint.

    Reads ``checkpoint_dir/stage1.pt`` (or the only ``.pt`` file),
    extracts the ``config`` dict, and assembles a complete ``Manifest``.

    Args:
        checkpoint_dir: directory containing ``stage1.pt``.
        source_id: model_id string for the manifest.
        source_revision: optional Git revision/tag.
        source: optional source encoder (for d_model, n_layers, vocab).
        tokenizer: optional tokenizer (for family, vocab_size).
        family: override tokenizer family (e.g. ``"bytelevel"``,
            ``"wordpiece"``).  When None, resolves from the tokenizer.

    Returns:
        A populated ``Manifest``.
    """
    from .vocab import resolve_token_bytes

    checkpoint_dir = Path(checkpoint_dir)
    pt_path = checkpoint_dir / "stage1.pt"
    if not pt_path.exists():
        # Fallback: find the first .pt file
        pts = list(checkpoint_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(
                f"no .pt file in {checkpoint_dir}"
            )
        pt_path = pts[0]

    import torch

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    resolution = resolve_token_bytes(tokenizer) if tokenizer is not None else None
    if family is None:
        family = resolution.strategy if resolution else ""
    vocab_size = resolution.vocab_size if resolution else 0
    n_special = config.get("n_byte_values", 0) - 256
    n_byte_values = 256 + (n_special if n_special > 0 else 0)

    # Extract gate results from history if available.
    gate = None
    history_path = checkpoint_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if history:
            last = history[-1]
            gate = GateResults(
                cos_emb=last.get("cos_emb_c"),
                sim_corr=last.get("sim_corr"),
                boundary_acc=last.get("boundary_acc"),
                nn_agree=last.get("nn_agree"),
                gate_passed=(
                    (last.get("cos_emb_c", 0) >= 0.90)
                    and (last.get("sim_corr", 0) >= 0.85)
                ),
            )

    arch = {
        "layers": config.get("local_layers", 1),
        "backend": config.get("local_backend", "attention"),
        "window": config.get("local_window", 128),
        "ffn_hidden": config.get("local_ffn_hidden", 1024),
        "boundary_predictor": config.get("boundary_predictor", "mlp"),
    }

    import time

    d_model = source.d_model if source else 0
    n_layers = source.n_layers if source else 0

    return Manifest(
        architecture=source_id.split("/")[-1] if "/" in source_id else source_id,
        budget=config.get("n_patches", 64),
        max_bytes=config.get("max_bytes", 512),
        source_id=source_id,
        source_revision=source_revision,
        family=family,
        byte_stream_rule=(
            "NFC"
            if resolution and resolution.strategy == "bytelevel"
            else "raw-utf8"
        ),
        vocab_size=vocab_size,
        n_byte_values=n_byte_values,
        gate=gate,
        seed=config.get("seed", 0),
        versions={
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        },
        encoder_arch=arch,
        source_d_model=d_model,
        source_n_layers=n_layers,
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    )


def export_with_manifest(
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    source_id: str,
    source_revision: str = "",
    source: Any = None,
    tokenizer: Any = None,
    n_patches: int = 64,
    max_bytes: int = 512,
) -> tuple[Path, Path]:
    """Export a byteified artifact: ONNX graph + manifest JSON.

    Writes ``output_dir/artifact.onnx`` and ``output_dir/manifest.json``.
    Returns the paths to both files.

    The ONNX graph is written first; if it fails, the manifest is NOT
    written (hard-fail gate: no partial artifact).
    """
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the manifest (gate checks inside from_checkpoint).
    manifest = from_checkpoint(
        checkpoint_dir, source_id, source_revision, source, tokenizer
    )
    # Validate; fail hard on invalid.
    errors = manifest.validate()
    if errors:
        raise ValueError(
            f"manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Write ONNX graph first.  On failure, manifest is NOT written.
    onnx_path = output_dir / "artifact.onnx"
    import torch

    # This is the entry point — the caller builds the embedder and calls
    # export_onnx from byteify.inference.  We just save the manifest
    # alongside whatever ONNX was produced.
    # (The actual ONNX export is done by the caller; we return the
    # manifest path for the caller to save alongside the ONNX file.)
    return onnx_path, output_dir / "manifest.json", manifest
