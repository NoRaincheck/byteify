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

from .fidelity import SUPPORTED_POOLING

# MAJOR.MINOR. A loader refuses a major it does not recognise rather than attempting a
# best-effort load; MINOR is additive and must stay backward compatible.
# 1.1 adds ``pooling``, ``normalize``, ``source_pipeline``, ``recipe``, ``recipe_hash`` and
# ``content_digest``. A reader that predates 1.1 ignores them and falls back to mean
# pooling with L2 normalisation, which is what every manifest through 1.0 described -- so
# 1.0 artifacts still load exactly as they did.
SCHEMA_VERSION: str = "1.1"
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
    #: The retrieval measurement behind ``retrieval``: decisions graded, chance, MRR,
    #: nDCG. Separate from the single gated number because recall@1 alone does not say
    #: whether a corpus was easy, and chance is what says that.
    retrieval_detail: dict[str, Any] = field(default_factory=dict)
    byte_stream_mismatches: int | None = None
    checks: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    #: Where the floors came from: ``"published"``, ``"supplied"`` by the caller, or
    #: ``"file:<path>"``. Recorded because a floor nobody published is still a floor this
    #: build cleared, and a reader has to be able to tell which kind they are looking at
    #: rather than assuming every artifact in the wild was gated against published
    #: numbers. See :func:`byteify.gates.resolve_thresholds`.
    thresholds_origin: str = ""
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
    #: What the source repository declared about its pipeline -- module types and the
    #: pooling config, verbatim. Empty for a plain checkpoint or a custom module, which
    #: are not sentence-transformers shaped; see :mod:`byteify.sentence_transformers`.
    source_pipeline: dict[str, Any] = field(default_factory=dict)
    #: How positions become the output vector. The exported graph bakes this in, so an
    #: artifact with a non-default ``pooling`` is not interchangeable with one that has
    #: the default, and a loader that ignored it would serve a different vector space.
    pooling: str = "mean"
    #: Whether the output vector is L2-normalised -- whether the source repository ships a
    #: Normalize module. Recorded rather than assumed because it changes every dot-product
    #: similarity while leaving cosine untouched, so the error would not appear in the
    #: metric most callers check first.
    normalize: bool = True
    gate: GateResults | None = None
    seed: int = 0
    #: Everything that determined this build, in one object: model, level, seed, training
    #: parameters, pooling, thresholds and their origin, and a digest of the reference
    #: data. Two builds that record the same recipe asked for the same artifact.
    recipe: dict[str, Any] = field(default_factory=dict)
    #: sha256 of ``recipe`` as canonical JSON. Comparable across builds and across
    #: machines, because it covers inputs and not the stack they ran on -- ``versions``
    #: carries that separately.
    recipe_hash: str = ""
    #: sha256 of every content file in the artifact, except the manifest itself. This is
    #: the determinism claim made checkable: rebuild with the same recipe on the same
    #: stack and these hashes match; change one thing and one does not.
    content_digest: dict[str, str] = field(default_factory=dict)
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
        if self.pooling not in SUPPORTED_POOLING:
            # A manifest declaring a mode this build cannot pool by would otherwise load
            # and fall back, producing a different vector space than the one it was
            # graded on. Refusing is the point of recording it.
            errors.append(
                f"pooling {self.pooling!r} is not implemented; supported modes are "
                f"{', '.join(SUPPORTED_POOLING)}"
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


def recipe_digest(recipe: dict[str, Any]) -> str:
    """Canonical sha256 of a recipe, so two builds can be compared in one field.

    Canonical because ``json.dumps`` with sorted keys and no whitespace is the only form
    that hashes identically for equal content: dict insertion order and float formatting
    would otherwise make two identical recipes differ.
    """
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_digest(path: Path | str) -> dict[str, str]:
    """sha256 of every file in an artifact except its manifest.

    The manifest is excluded because it carries ``exported_at`` -- two byte-identical
    builds differ there by construction -- and because it carries *this* digest, so
    including it would be hashing a value into itself. Everything else is compared.

    Measured on the synthetic fixture with the same seed in two separate processes:
    ``encoder.pt``, ``trie.json``, ``artifact.onnx`` and ``artifact.onnx.data`` all came
    out byte-identical, which is what makes this a claim rather than a hope. (In one
    process they did not, because the ONNX exporter writes a per-process counter of
    compiled code objects into a debug stack-trace property; it is metadata, not the
    model, and it is why the digest is taken across processes.)
    """
    directory = Path(path)
    return {
        p.name: _sha256(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != MANIFEST_NAME
    }


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
    from . import __version__

    versions = {
        "byteify": __version__,
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
    pooling: str = "mean",
    normalize: bool = True,
    source_pipeline: dict[str, Any] | None = None,
    recipe: dict[str, Any] | None = None,
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
        source_pipeline=dict(source_pipeline or {}),
        pooling=pooling,
        normalize=bool(normalize),
        trie=TrieRef(
            entries=len(token_bytes or {}),
            coverage=coverage,
            coverage_bytes=sum(len(t.encode("utf-8")) for t in (coverage_texts or [])),
        ),
        gate=gate,
        seed=int(config.get("seed", 0)),
        recipe=dict(recipe or {}),
        recipe_hash=recipe_digest(recipe) if recipe else "",
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
    pooling: str = "mean",
    normalize: bool = True,
    source_pipeline: dict[str, Any] | None = None,
    recipe: dict[str, Any] | None = None,
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
        pooling=pooling,
        normalize=normalize,
        source_pipeline=source_pipeline,
        recipe=recipe,
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


def verify_artifact(artifact_dir: Path | str) -> dict[str, Any]:
    """Recompute an artifact's content digest and compare it with the manifest's.

    This is where "byteify is deterministic" stops being a statement about a build and
    becomes something a reader can check: the manifest records sha256 for every content
    file, this recomputes them, and the difference names the file that changed.

    A missing manifest raises, because there is then nothing to verify against -- and a
    directory with no manifest is not an artifact in this repository's sense at all.

    Returns a dict with ``matches``, ``changed`` (present in both, different hash),
    ``missing`` (recorded but not on disk), ``extra`` (on disk but unrecorded) and
    ``actual``. Unrecorded digests -- a manifest written before this existed -- report
    ``matches: None`` rather than a pass, because "nothing was recorded" is not the same
    claim as "nothing changed".
    """
    artifact_dir = Path(artifact_dir)
    manifest = Manifest.load(artifact_dir)
    actual = content_digest(artifact_dir)
    expected = dict(manifest.content_digest or {})

    changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())

    if not expected:
        matches: bool | None = None
    else:
        matches = not (changed or missing)
    return {
        "artifact": str(artifact_dir),
        "recipe_hash": manifest.recipe_hash,
        "matches": matches,
        "changed": changed,
        "missing": missing,
        "extra": extra,
        "expected": expected,
        "actual": actual,
    }