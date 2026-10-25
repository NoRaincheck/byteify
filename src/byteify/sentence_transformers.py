"""Reading a sentence-transformers repository's declared pipeline.

A sentence-transformers model is not a checkpoint. It is a *pipeline* described by
``modules.json`` -- a Transformer, then a Pooling, then optionally a Normalize -- and
``model.encode("...")`` returns the output of that whole pipeline. Byteify's promise is
that the byteified model behaves like the model it came from, so "the model it came from"
has to mean the pipeline and not just its transformer body.

This module reads that declaration rather than inferring it. Byteify's own default is a
masked mean pool followed by L2 normalisation, which coincides with a mean-pooled model
that ships a Normalize module -- and differs, silently, from everything else:

* a repository that pools by CLS would be graded against a mean pool of the same text, so
  the fidelity gate would measure two different quantities and call the gap "fidelity";
* a repository with no Normalize module would be *served* with a unit norm its source
  never had, which changes every dot-product similarity while leaving cosine untouched,
  so the error would not show up in the metric most people check.

No ``sentence_transformers`` dependency is introduced: this reads the repository format,
which is plain JSON on disk. A repository with no ``modules.json``, or one whose
``modules.json`` declares a custom module rather than a Pooling, is not sentence-transformers
shaped -- ``hotchpotch/bekko-system-one-v0-17m`` is the second case -- and is reported as
undeclared so the caller keeps byteify's existing behaviour instead of guessing.

**What is refused, and why.** Two declarations are read but not implemented: a ``Dense``
projection and the ``weightedmean_tokens`` pooling mode. Both would put a transform
between the encoder and the embedding that byteify would have to reproduce without being
able to check it does -- ``weightedmean`` needs per-token attention weights, which the
patch axis does not carry. Saying which file said it is the alternative to silently
pooling by mean and reporting a number about a model nobody will deploy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fidelity import SUPPORTED_POOLING

#: Modules a repository may declare that byteify implements.
SUPPORTED_MODULE_TYPES = ("Transformer", "Pooling", "Normalize")

#: What the pipeline falls back to when a repository declares nothing. Byteify's own
#: default, and the behaviour every existing artifact was built with.
DEFAULT_POOLING = "mean"
DEFAULT_NORMALIZE = True

#: Files fetched when a repository is resolved by id. Small on purpose: this runs before
#: any training, and the weights are fetched separately by :mod:`byteify.hub`.
REPO_PATTERNS = (
    "modules.json",
    "config.json",
    "config_sentence_transformers.json",
    "*/config.json",
    "*/config_sentence_transformers.json",
)

#: The keys in ``1_Pooling/config.json`` that name a mode, in the order they are checked
#: so an ambiguous config resolves deterministically rather than by dict order.
POOLING_KEYS = (
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_mean_tokens", "mean"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt"),
    ("pooling_mode_lasttoken", "lasttoken"),
    ("pooling_mode_weightedmean_tokens", "weightedmean"),
    ("pooling_mode_prompt_tokens", "prompt"),
)


class UnsupportedSentenceTransformer(ValueError):
    """The repository declares a pipeline byteify does not reproduce, naming the file."""


@dataclass(frozen=True)
class Pipeline:
    """What a repository says about how it turns tokens into an embedding."""

    #: False when the repository is not sentence-transformers shaped, in which case every
    #: other field is byteify's default and must not be read as a declaration.
    declared: bool = False
    pooling: str = DEFAULT_POOLING
    normalize: bool = DEFAULT_NORMALIZE
    #: Module types as declared, e.g. ``("Transformer", "Pooling", "Normalize")``.
    modules: tuple[str, ...] = ()
    #: The pooling module's own config, verbatim, for the manifest's provenance.
    pooling_config: dict[str, Any] = field(default_factory=dict)
    #: Which file the mode was read from, for an error message that points somewhere real.
    source: str = ""

    @property
    def is_sentence_transformer(self) -> bool:
        return self.declared

    def as_manifest(self) -> dict[str, Any]:
        """This declaration as a manifest records it, empty when nothing was declared.

        Empty rather than a filled-in default, because the difference between "the
        repository said mean" and "there was nothing to read" is exactly what a reader
        needs, and both would otherwise appear as ``pooling: "mean"``.
        """
        if not self.declared:
            return {}
        return {
            "modules": list(self.modules),
            "pooling": self.pooling,
            "normalize": self.normalize,
            "pooling_config": self.pooling_config,
            "source": self.source,
        }

    def describe(self) -> str:
        if not self.declared:
            return "not a sentence-transformers pipeline (byteify's default: mean + L2)"
        parts = [f"pooling={self.pooling}", "normalize" if self.normalize else "no-normalize"]
        return f"sentence-transformers pipeline: {', '.join(parts)}"


def read_pipeline(model_or_name: str | Path) -> Pipeline:
    """Read how a repository pools, and whether it normalises.

    Raises :class:`UnsupportedSentenceTransformer` when the repository *is* declared and
    uses something byteify cannot reproduce, and returns an undeclared pipeline when it is
    not sentence-transformers shaped at all. The distinction matters: the first is a model
    byteify refuses, the second is one it already byteifies by its existing route.
    """
    repository = _repository(model_or_name)
    modules_file = repository / "modules.json"
    if not modules_file.exists():
        return Pipeline(source=str(modules_file))

    modules = _read_modules(modules_file)
    types = tuple(_type_name(m) for m in modules)
    if "Pooling" not in types:
        # A custom module -- `hotchpotch/bekko-system-one-v0-17m` declares
        # `inference_v0.BekkoInference` alone. Not a pipeline byteify should re-derive.
        return Pipeline(modules=types, source=str(modules_file))

    _refuse_dense(modules_file, types)
    mode, config, pooling_file = _read_pooling(repository, modules)
    if mode == "weightedmean":
        raise UnsupportedSentenceTransformer(
            f"{pooling_file} declares `weightedmean_tokens`, which weights each token by "
            f"the model's attention scores. The byteified student axis has patches, not "
            f"attention weights over the source's tokens, so this cannot be reproduced. "
            f"Refused rather than pooled by mean, which would report fidelity for a "
            f"model this repository never produced."
        )
    if mode == "prompt":
        raise UnsupportedSentenceTransformer(
            f"{pooling_file} declares `pooling_mode_prompt_tokens`, which pools prompt and "
            f"document positions separately. The byte axis carries no prompt boundary -- it "
            f"is one run of bytes -- so the split cannot be reproduced."
        )
    if config.get("include_prompt") is False:
        raise UnsupportedSentenceTransformer(
            f"{pooling_file} sets `include_prompt: false`, which excludes prompt tokens "
            f"from the pooled vector. The byte axis carries no prompt boundary, so the "
            f"exclusion cannot be reproduced."
        )
    if mode not in SUPPORTED_POOLING:
        raise UnsupportedSentenceTransformer(
            f"{pooling_file} declares pooling mode {mode!r}, which byteify does not "
            f"implement; supported modes are {', '.join(SUPPORTED_POOLING)}"
        )

    return Pipeline(
        declared=True,
        pooling=mode,
        normalize="Normalize" in types,
        modules=types,
        pooling_config=config,
        source=str(pooling_file),
    )


def _repository(model_or_name: str | Path) -> Path:
    """A local directory for the repository, from disk or from the hub cache."""
    path = Path(model_or_name)
    if os.path.isdir(path):
        return path

    from huggingface_hub import snapshot_download

    # Resolves against the cache first, so a repository fetched once loads with
    # HF_HUB_OFFLINE=1 -- how the test suite and a serving host both run. The pooling
    # config is deliberately in the pattern list: `hub.LAYOUT_PATTERNS` fetches weights
    # and tokenizer files, and does not know a Pooling module's config is needed.
    #
    # A failure here is not converted to "no pipeline". It has to be an error: silently
    # reading an unreadable repository as undeclared would pool an ST model by byteify's
    # default and report fidelity for a different embedding than the source produces.
    try:
        return Path(snapshot_download(str(model_or_name), allow_patterns=list(REPO_PATTERNS)))
    except OSError as exc:
        raise FileNotFoundError(
            f"could not read {model_or_name}'s repository to see how it pools: {exc}\n"
            "It is not in the local HuggingFace cache. Download it first (unset "
            "HF_HUB_OFFLINE), or pass a local directory path."
        ) from exc


def _read_modules(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnsupportedSentenceTransformer(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise UnsupportedSentenceTransformer(
            f"{path} is not a list of modules; sentence-transformers writes a JSON array"
        )
    return [m for m in raw if isinstance(m, dict)]


def _type_name(module: dict[str, Any]) -> str:
    """``sentence_transformers.models.Pooling`` -> ``Pooling``.

    The suffix rather than the whole string because the package moved between
    ``sentence_transformers.models.X`` and bare ``X`` across releases, and both appear in
    published repositories.
    """
    return str(module.get("type", "")).rsplit(".", 1)[-1]


def _refuse_dense(modules_file: Path, types: tuple[str, ...]) -> None:
    if "Dense" not in types:
        return
    raise UnsupportedSentenceTransformer(
        f"{modules_file} declares a `Dense` projection between the encoder and the "
        f"embedding. Its weights would have to ship inside the artifact and be applied in "
        f"the exported graph, and no repository carrying one has been measured here, so "
        f"the format is not verified. Refused rather than approximated: applying a dense "
        f"layer byteify did not reproduce would produce embeddings in a different space "
        f"from the source's while every gate still passed."
    )


def _read_pooling(
    repository: Path, modules: list[dict[str, Any]]
) -> tuple[str, dict[str, Any], Path]:
    """The declared pooling mode, its config file, and where they came from."""
    path = next(
        (str(m.get("path", "")) for m in modules if _type_name(m) == "Pooling"), ""
    )
    pooling_dir = repository / path
    candidates = [
        pooling_dir / "config.json",
        pooling_dir / "config_sentence_transformers.json",
        repository / "config_sentence_transformers.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            config = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UnsupportedSentenceTransformer(f"{candidate} is not valid JSON: {exc}") from exc
        mode = _mode_from(config)
        if mode:
            return mode, config, candidate
        # The root `config_sentence_transformers.json` often holds only __version__ and
        # prompts, so keep looking rather than reading "no mode key" as "no mode".
    raise UnsupportedSentenceTransformer(
        f"{pooling_dir} carries no pooling config naming a mode "
        f"({', '.join(k for k, _ in POOLING_KEYS)}); byteify will not assume one, because "
        f"pooling the wrong way reports fidelity for a model this repository never produced"
    )


def _mode_from(config: dict[str, Any]) -> str:
    """The single mode this config declares. Empty when it declares none.

    Ambiguity is refused rather than resolved by preference: two modes set true is a
    repository byteify does not understand, and picking one would silently choose a
    different embedding than the source's.
    """
    declared = [mode for key, mode in POOLING_KEYS if config.get(key)]
    if not declared:
        return ""
    if len(declared) > 1:
        raise UnsupportedSentenceTransformer(
            f"the pooling config declares {len(declared)} modes at once "
            f"({', '.join(declared)}); byteify needs exactly one to know what it is "
            f"reproducing"
        )
    return declared[0]
