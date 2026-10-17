"""The byteified artifact manifest: an artifact's sole entry point.

An artifact is a directory::

    artifact/
        manifest.json     # the contract
        artifact.onnx     # the graph
        artifact.onnx.data  # external weights, if the graph spilled
        trie.json         # id -> byte string, so inference needs no source tokenizer
        encoder.pt        # encoder weights, for the Python inference path

The manifest is the only thing a loader needs. It carries the local-encoder architecture,
the patch budget, the byte-axis length, the source model and revision, the tokenizer
family and byte-stream rule, the resolved vocabulary and byte-value counts, the suffix
trie's measured coverage, the gate results with the thresholds used, the seed, and library
versions.

Why it is derived rather than declared
--------------------------------------
Every dimension in here was, at one point, read from a *guessed* key or defaulted to a
constant, and each guess was wrong in a way that only showed up later:

* ``n_byte_values`` came from a config key the trainers never wrote, so a 263-row byte
  table was described as 256 and a loader built from it hit a shape error.
* ``ffn_hidden`` was read as ``local_ffn_hidden`` while the trainers write ``local_ffn``,
  so it silently fell back to the default for every checkpoint.
* the patch budget was not in any checkpoint at all, so it was always reported as the
  default regardless of what the artifact was actually exported with.

So the dimensions that *can* be read from the weights are read from the weights, the ones
that are an export-time decision are passed in rather than defaulted, and the ones that
cannot be known without the source model are required rather than guessed. A manifest that
cannot be filled in honestly is not written.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# MAJOR.MINOR. A loader refuses a major it does not recognise rather than attempting a
# best-effort load; MINOR is additive and must stay backward compatible.
SCHEMA_VERSION: str = "1.0"
SCHEMA_MAJOR: int = 1

MANIFEST_NAME = "manifest.json"
GRAPH_NAME = "artifact.onnx"
TRIE_NAME = "trie.json"
ENCODER_NAME = "encoder.pt"


class ManifestError(ValueError):
    """A manifest could not be read, or describes something that cannot be served."""


class UnsupportedSchemaVersion(ManifestError):
    """The manifest's major version is not one this loader implements."""


def _parse_version(text: str) -> tuple[int, int]:
    parts = text.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ManifestError(f"schema version {text!r} is not MAJOR.MINOR")
    return int(parts[0]), int(parts[1])


def check_schema_version(text: str) -> tuple[int, int]:
    """Reject a manifest whose major version this loader does not implement.

    A best-effort load of a newer major would build an encoder from fields that may have
    changed meaning, and the failure would surface as a shape error far from its cause.
    """
    major, minor = _parse_version(text)
    if major != SCHEMA_MAJOR:
        raise UnsupportedSchemaVersion(
            f"manifest schema version {text} is not supported; this byteify reads "
            f"major version {SCHEMA_MAJOR}. Refusing rather than loading it "
            f"approximately."
        )
    return major, minor


@dataclass(frozen=True, slots=True)
class TrieRef:
    """Where the artifact's suffix trie lives and how well it covers real text.

    ``coverage`` is the measured fraction of byte positions in reference text that the
    trie resolves to a vocabulary token. It is recorded rather than assumed because it
    varies sharply by tokenizer family -- complete for byte-level BPE, much lower for
    WordPiece and SentencePiece -- and it is the honest way to predict an artifact's
    quality before committing to it.
    """

    path: str = TRIE_NAME
    sha256: str = ""
    entries: int = 0
    coverage: float | None = None
    coverage_bytes: int = 0


@dataclass(frozen=True, slots=True)
class GateResults:
    """Numerical quality gates, the thresholds used, and what the level claims.

    The level and its guarantee live here rather than beside it: a reviewer reading an
    artifact needs to see what it promised and whether it held in one place, and a field
    that is not in this dataclass is a field that is silently dropped on the way to disk.
    """

    cos_emb: float | None = None
    sim_corr: float | None = None
    boundary_acc: float | None = None
    nn_agree: float | None = None
    onnx_max_abs_diff: float | None = None
    outputs_finite: float | None = None
    retrieval: float | None = None
    byte_stream_mismatches: int | None = None
    checks: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    level: str = ""
    guarantee: str = ""
    declared: dict[str, Any] = field(default_factory=dict)
    fidelity: dict[str, Any] = field(default_factory=dict)
    gate_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateResults:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to load and serve one byteified artifact."""

    architecture: str = ""
    budget: int = 0
    max_bytes: int = 0
    source_id: str = ""
    source_revision: str = ""
    family: str = ""
    byte_stream_rule: str = ""
    vocab_size: int = 0
    n_byte_values: int = 0
    encoder_arch: dict[str, Any] = field(default_factory=dict)
    source_d_model: int = 0
    source_n_layers: int = 0
    trie: TrieRef = field(default_factory=TrieRef)
    byte_stream: dict[str, Any] = field(default_factory=dict)
    gate: GateResults | None = None
    seed: int = 0
    versions: dict[str, str] = field(default_factory=dict)
    exported_at: str = ""
    schema_version: str = SCHEMA_VERSION

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trie"] = asdict(self.trie)
        d["gate"] = self.gate.to_dict() if self.gate is not None else None
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        check_schema_version(str(d.get("schema_version", "")))
        gate = GateResults.from_dict(d["gate"]) if d.get("gate") else None
        trie = TrieRef(**d["trie"]) if d.get("trie") else TrieRef()
        fields = {
            k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k not in {"gate", "trie"}
        }
        return cls(gate=gate, trie=trie, **fields)

    def stream_rule(self):
        """The declared byte-stream rule, as an object.

        Raises rather than guessing when the manifest does not carry one: a loader that
        invented a rule would produce a byte stream that differs from the trained one,
        which is the failure ADR 0011 exists to prevent.
        """
        from .bytestream import ByteStreamRule

        if self.byte_stream:
            return ByteStreamRule.from_dict(self.byte_stream)
        raise ManifestError(
            "manifest carries no byte_stream rule; inference inputs cannot be built "
            "without one, and guessing it would produce a different stream than training"
        )

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(d, dict):
            raise ManifestError("manifest is not a JSON object")
        return cls.from_dict(d)

    def write_to(self, path: Path) -> None:
        _atomic_write(path, self.to_json())

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path | str) -> Manifest:
        """Read a manifest, rejecting one this loader cannot faithfully interpret."""
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"no manifest at {path}")
        manifest = cls.from_json(path.read_text(encoding="utf-8"))
        errors = manifest.validate()
        if errors:
            raise ManifestError(
                "manifest is not loadable:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return manifest

    @classmethod
    def load(cls, artifact_dir: Path | str) -> Manifest:
        """Load the manifest from an artifact directory."""
        return cls.from_file(Path(artifact_dir) / MANIFEST_NAME)

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return the reasons this manifest cannot be served. Empty means loadable."""
        errors: list[str] = []
        if not self.source_id:
            errors.append("source_id is empty")
        if not self.family:
            errors.append("family is empty")
        if not self.byte_stream_rule:
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
        if self.n_byte_values < 256:
            errors.append(
                f"n_byte_values must be at least 256 (real bytes); got {self.n_byte_values}"
            )
        if not self.byte_stream:
            errors.append(
                "byte_stream is empty: inference inputs cannot be built from this manifest"
            )
        for key in ("layers", "backend", "window", "ffn_hidden", "boundary_predictor"):
            if key not in self.encoder_arch:
                errors.append(f"encoder_arch is missing {key!r}")
        if not self.trie.path:
            errors.append("trie.path is empty: inference would need the source tokenizer")
        if self.gate is not None and not self.gate.gate_passed:
            errors.append("recorded gate result is a failure; this artifact was not cleared")
        return errors


# -- writing -------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a crash cannot leave a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        # mkstemp creates 0600. An artifact file only its owner can read is not an
        # artifact anyone else can serve, so fall back to the process umask.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_trie(path: Path, token_bytes: dict[int, bytes]) -> str:
    """Serialise id -> byte string, and return its sha256.

    The trie itself is a derived structure; the vocabulary byte strings are the fact. They
    are stored so that a loader rebuilds the trie with byteify's own code rather than
    trusting a serialised tree layout to stay compatible.
    """
    payload = {str(i): b.hex() for i, b in sorted(token_bytes.items()) if b}
    _atomic_write(path, json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return _sha256(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trie(path: Path) -> dict[int, bytes]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): bytes.fromhex(v) for k, v in payload.items()}


def measure_trie_coverage(token_bytes: dict[int, bytes], texts: list[str]) -> float:
    """Fraction of byte positions in ``texts`` the trie resolves to a vocabulary token.

    Measured on raw UTF-8 because that is the stream inference will see. A position that
    resolves to nothing is one where ``T_SubwordSuffix`` contributes no pretrained vector,
    which is exactly the degradation the number is meant to predict.
    """
    from .expand import ByteTrie

    trie = ByteTrie(token_bytes)
    total = 0
    covered = 0
    for text in texts:
        stream = list(text.encode("utf-8"))
        if not stream:
            continue
        for i in range(len(stream)):
            total += 1
            if trie.longest_suffix_id(stream, i) >= 0:
                covered += 1
    return covered / total if total else 0.0


# -- building ------------------------------------------------------------


def library_versions() -> dict[str, str]:
    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
    }
    for module in ("torch", "transformers", "onnx", "onnxruntime", "numpy"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            # An optional dependency that is absent is simply not recorded; a manifest
            # should still say which version of torch built it.
            continue
    return versions


def manifest_from_checkpoint(
    checkpoint: Path | str,
    *,
    source: Any,
    budget: int,
    max_bytes: int,
    rule: Any,
    family: str = "",
    source_revision: str = "",
    token_bytes: dict[int, bytes] | None = None,
    coverage_texts: list[str] | None = None,
    gate: GateResults | None = None,
) -> Manifest:
    """Build a manifest for an encoder checkpoint.

    Dimensions that the weights already state are read from the weights, because reading
    them from a config key is how the byte table came to be described as 256 rows when it
    held 263. Dimensions that are an export-time decision are arguments, not defaults.
    Anything that cannot be known without the source model is taken from ``source``,
    which is therefore required.

    ``budget`` and ``max_bytes`` have no defaults on purpose. The patch budget is the axis
    the whole serving guarantee rests on (ADR 0002); defaulting it would produce a
    manifest that quietly misdescribes the artifact.
    """
    import torch

    from .source import LocalEncoderSpec

    path = Path(checkpoint)
    if not path.exists():
        raise ManifestError(f"no checkpoint at {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder")
    if not state:
        raise ManifestError(f"{path} holds no 'encoder' state dict")

    required = ("embed.byte.weight", "embed.suffix.weight")
    missing = [k for k in required if k not in state]
    if missing:
        raise ManifestError(f"{path} is missing {missing}; it is not a byteified encoder")

    n_byte_values = int(state["embed.byte.weight"].shape[0])
    subword_vocab_size = int(state["embed.suffix.weight"].shape[0])
    d_model = int(state["embed.byte.weight"].shape[1])

    config = ckpt.get("config") or ckpt.get("args") or {}
    spec = LocalEncoderSpec.from_config(config)

    if source is None:
        raise ManifestError(
            "a source encoder is required: the layer count and patch budget cannot be "
            "recovered from the byteified weights alone"
        )

    coverage = None
    if coverage_texts:
        coverage = measure_trie_coverage(token_bytes or {}, coverage_texts)

    return Manifest(
        architecture=source_id_to_name(source, config),
        budget=int(budget),
        max_bytes=int(max_bytes),
        source_id=source.model_id,
        source_revision=source_revision,
        family=family or getattr(source, "family", ""),
        byte_stream_rule=rule.rule,
        vocab_size=subword_vocab_size,
        n_byte_values=n_byte_values,
        encoder_arch=spec.as_dict(),
        source_d_model=int(getattr(source, "d_model", d_model)),
        source_n_layers=int(getattr(source, "n_layers", 0)),
        byte_stream=rule.to_dict(),
        trie=TrieRef(
            entries=len(token_bytes or {}),
            coverage=coverage,
            coverage_bytes=sum(len(t.encode("utf-8")) for t in (coverage_texts or [])),
        ),
        gate=gate,
        seed=int(config.get("seed", 0)),
        versions=library_versions(),
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def source_id_to_name(source: Any, config: dict[str, Any]) -> str:
    name = getattr(source, "model_id", "") or ""
    if name:
        return name.split("/")[-1]
    return str(config.get("model", ""))


# -- the artifact --------------------------------------------------------


def export_artifact(
    embedder: Any,
    tokenizer: Any,
    checkpoint: Path | str,
    output_dir: Path | str,
    *,
    budget: int,
    max_bytes: int,
    source_revision: str = "",
    gate: GateResults | None = None,
    coverage_texts: list[str] | None = None,
    graph_name: str = GRAPH_NAME,
    example_batch: tuple[Any, Any, Any] | None = None,
) -> Manifest:
    """Write a complete artifact: graph, trie, encoder weights, manifest.

    The manifest is written **last**. Until it exists the directory is not an artifact,
    so a failure part-way through leaves something a loader will refuse rather than
    something that loads and computes the wrong thing.

    ``example_batch`` fixes the byte axis the graph is traced at. The graph's byte axis is
    static (ADR 0002), so a caller that intends to pad to ``max_bytes`` must trace at that
    width -- otherwise the graph bakes in whatever width its example happened to be.
    """
    from .bytestream import rule_for_family
    from .inference import export_onnx
    from .vocab import resolve_token_bytes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / graph_name
    export_onnx(
        embedder,
        tokenizer,
        path=str(graph_path),
        max_bytes=max_bytes,
        n_patches=budget,
        example_batch=example_batch,
    )

    resolution = resolve_token_bytes(tokenizer)
    token_bytes = {i: b for i, b in enumerate(resolution.token_bytes) if b}
    digest = write_trie(output_dir / TRIE_NAME, token_bytes)

    from .alignment import discover_special_bytes

    specials = discover_special_bytes(tokenizer)
    rule = rule_for_family(
        resolution.strategy,
        n_byte_values=256 + len(specials),
        special_tokens=specials,
    )

    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.save(ckpt["encoder"], output_dir / ENCODER_NAME)

    manifest = manifest_from_checkpoint(
        checkpoint,
        source=embedder.source,
        budget=budget,
        max_bytes=max_bytes,
        rule=rule,
        family=resolution.strategy,
        source_revision=source_revision,
        token_bytes=token_bytes,
        coverage_texts=coverage_texts,
        gate=gate,
    )
    manifest = replace(manifest, trie=replace(manifest.trie, sha256=digest))

    errors = manifest.validate()
    if errors:
        raise ManifestError(
            "refusing to write an artifact whose manifest does not describe it:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    manifest.write_to(output_dir / MANIFEST_NAME)
    return manifest


def load_artifact(
    artifact_dir: Path | str,
    *,
    device: str = "cpu",
) -> tuple[Manifest, Any]:
    """Load an artifact through its manifest alone.

    Returns ``(manifest, embedder)``. The source *model* is fetched by id because the
    byteified encoder delegates to it; the source *tokenizer* is not needed, because the
    artifact ships the vocabulary byte strings its suffix trie is built from.
    """
    from .loader import build_embedder_from_manifest

    artifact_dir = Path(artifact_dir)
    manifest = Manifest.load(artifact_dir)

    trie_path = artifact_dir / manifest.trie.path
    if not trie_path.exists():
        raise ManifestError(
            f"artifact references {manifest.trie.path} but it is not in {artifact_dir}; "
            f"without it inference would need the source tokenizer"
        )
    if manifest.trie.sha256:
        actual = _sha256(trie_path)
        if actual != manifest.trie.sha256:
            raise ManifestError(
                f"{manifest.trie.path} does not match the sha256 recorded in the manifest "
                f"({actual[:12]}... vs {manifest.trie.sha256[:12]}...)"
            )

    embedder = build_embedder_from_manifest(manifest, artifact_dir, device=device)
    return manifest, embedder