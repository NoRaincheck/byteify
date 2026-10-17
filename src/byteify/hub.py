"""Resolving a source model's files when the repository is not a plain checkpoint.

Most published embedding models are plain checkpoints: ``config.json``, weights and a
tokenizer at the repository root, which ``AutoModel.from_pretrained`` and
``AutoTokenizer.from_pretrained`` read without help. Not all of them are.

A sentence-transformers repository whose module is custom code keeps the encoder inside a
module directory named by ``modules.json``, under a config file that is not called
``config.json``, with the encoder's weights carrying the module's own prefix and the task
heads stored beside it. ``hotchpotch/bekko-system-one-v0-17m`` is one: its encoder is
ModernBERT under ``0_BekkoInference/``, configured by ``backbone_config.json``, with keys
prefixed ``encoder.backbone.`` and three task heads (``choice``, ``noul``, ``score``) that
are not part of any encoder.

``AutoModel`` cannot read that layout, and that matters more than it looks: a byteified
encoder delegates its patch representations to the frozen source encoder, so level 1's
fidelity guarantee *is* that delegation and the source model is re-fetched every time an
artifact is loaded. A repository byteify can build from but cannot serve is not a supported
target, so both layouts resolve here to the same two things -- a model and a tokenizer --
and name which layout was found. "We guessed" is not an acceptable answer to "where did
these weights come from".

The fallback is a fallback: a plain repository still goes through ``AutoModel``, so this
adds a route rather than replacing one, and the ordinary path is unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The file names that mean "this directory holds an encoder", in the order they are tried.
#: ``config.json`` is the plain-checkpoint name and is not here: reaching this module at
#: all means ``AutoModel`` already failed on it.
CONFIG_NAMES = ("backbone_config.json", "config_sentence_transformers.json")

#: Weight prefixes stripped, longest first, when mapping a module's checkpoint onto a
#: ``nn.Module``. A custom module names its own attributes, and there is no way to know in
#: advance which one, so this is a declared list rather than a heuristic: a key that does
#: not start with one of these is reported as dropped, not silently re-homed.
WEIGHT_PREFIXES = ("encoder.backbone.", "encoder.", "backbone.", "transformer.", "model.")

#: Everything a fallback layout is allowed to fetch. Small on purpose: this runs on every
#: artifact load, so it must not pull a repository's ONNX exports and evaluation scripts.
LAYOUT_PATTERNS = (
    "modules.json",
    "*/modules.json",
    "**/backbone_config.json",
    "**/model.safetensors",
    "**/pytorch_model.bin",
    "**/tokenizer/tokenizer.json",
    "**/tokenizer/tokenizer_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


#: What ``AutoModel``/``AutoTokenizer`` raise for a repository they cannot read. Both are
#: needed and the difference is not a detail: a *missing* file is an ``OSError`` from the
#: hub, while a directory that exists and holds no ``config.json`` is a ``ValueError``
#: naming the missing key. Catching only the first meant a module-layout repository -- the
#: whole case this module exists for -- never reached the fallback at all.
LOAD_FAILURES = (OSError, ValueError)


@dataclass(frozen=True)
class ResolvedSource:
    """A source model, and the provenance of the files it was built from."""

    model: Any
    #: ``"checkpoint"`` for a plain ``AutoModel`` load, ``"module"`` for a module layout.
    layout: str
    #: The module directory inside the repository, empty for a plain checkpoint.
    module_path: str = ""
    #: Weights in the checkpoint that no encoder parameter claimed -- the task heads.
    dropped: tuple[str, ...] = field(default=())


def resolve_source_model(model_or_name) -> ResolvedSource:
    """Load a source encoder from a hub id, a local directory, or an in-memory model."""
    if not isinstance(model_or_name, str):
        return ResolvedSource(model=model_or_name, layout="memory")

    from transformers import AutoModel

    try:
        return ResolvedSource(model=AutoModel.from_pretrained(model_or_name), layout="checkpoint")
    except LOAD_FAILURES as plain_error:
        try:
            return _load_module_layout(model_or_name)
        except LOAD_FAILURES as layout_error:
            raise FileNotFoundError(
                f"could not load {model_or_name}: {plain_error}\n"
                f"It is not a plain checkpoint and not a readable module layout either: "
                f"{layout_error}\n"
                "If it is not in the local HuggingFace cache, download it first (unset "
                "HF_HUB_OFFLINE). A local directory path also works."
            ) from layout_error


def resolve_source_tokenizer(model_or_name):
    """The tokenizer that goes with a source model, for either repository layout.

    Paired with :func:`resolve_source_model` because they answer the same question about
    two files, and a build that resolves the weights through one route and the vocabulary
    through another is a build whose byte stream and its supervision disagree.
    """
    if not isinstance(model_or_name, str):
        raise TypeError(
            "resolve_source_tokenizer takes a model id or directory; an in-memory model "
            "carries no tokenizer with it"
        )

    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_or_name)
    except LOAD_FAILURES:
        repository = _repository(model_or_name)
        directory = _module_dir(repository)
        tokenizer_dir = directory / "tokenizer"
        tokenizer_file = tokenizer_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            tokenizer_file = directory / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"{model_or_name} has no root tokenizer and no module layout carrying one; "
                "byteify needs the tokenizer to resolve the vocabulary to byte strings"
            )

        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file), **_tokenizer_fields(tokenizer_file)
        )


# -- layout resolution -----------------------------------------------------


def _repository(model_or_name: str) -> Path:
    """A local directory for the repository, from disk or from the hub cache."""
    if os.path.isdir(model_or_name):
        return Path(model_or_name)
    from huggingface_hub import snapshot_download

    # `snapshot_download` resolves against the cache first, so a repository fetched once
    # loads with HF_HUB_OFFLINE=1 -- which is how the test suite and a serving host both
    # run.
    return Path(snapshot_download(model_or_name, allow_patterns=list(LAYOUT_PATTERNS)))


def _module_dir(repository: Path) -> Path:
    """The directory holding a module layout's encoder, or the repository root.

    ``modules.json`` names the module directories; the first one carrying a readable config
    wins. Repositories that are already a plain layout have no ``modules.json``, and their
    root is the answer.
    """
    modules_file = repository / "modules.json"
    if modules_file.exists():
        try:
            modules = json.loads(modules_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FileNotFoundError(f"{modules_file} is not valid JSON: {exc}") from exc
        candidates = [
            repository / str(module.get("path", ""))
            for module in modules
            if isinstance(module, dict) and module.get("path")
        ]
        for candidate in candidates:
            if any((candidate / name).exists() for name in CONFIG_NAMES):
                return candidate
        raise FileNotFoundError(
            f"no module in {modules_file} carries a config "
            f"({', '.join(CONFIG_NAMES)}); byteify cannot tell which one holds the encoder"
        )
    return repository


def _load_module_layout(model_or_name: str) -> ResolvedSource:
    """Build the encoder from a module layout: its config, its weights, no guesswork."""
    from transformers import AutoConfig, AutoModel

    repository = _repository(model_or_name)
    directory = _module_dir(repository)
    config_file = next(
        (directory / name for name in CONFIG_NAMES if (directory / name).exists()), None
    )
    if config_file is None:
        raise FileNotFoundError(f"{directory} carries no encoder config ({', '.join(CONFIG_NAMES)})")
    config = AutoConfig.from_pretrained(str(config_file))
    model = AutoModel.from_config(config)

    state = _load_weights(directory)
    remapped, dropped = _remap_weights(state, model)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # A missing parameter means the weights on disk and the architecture the config
    # describes disagree, and loading the overlap would hide that: the model would run,
    # and half of it would be random. Refuse instead.
    if missing or unexpected:
        raise FileNotFoundError(
            f"the checkpoint in {directory} does not match the encoder {config_file} "
            f"describes: {len(missing)} parameter(s) missing ({sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected ({sorted(unexpected)[:3]})"
        )

    return ResolvedSource(
        model=model,
        layout="module",
        module_path=str(directory.relative_to(repository)),
        dropped=tuple(dropped),
    )


def _load_weights(directory: Path) -> dict[str, Any]:
    import torch

    safetensors = directory / "model.safetensors"
    if safetensors.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors))
    torch_weights = directory / "pytorch_model.bin"
    if torch_weights.exists():
        return torch.load(str(torch_weights), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{directory} carries no encoder weights")


def _remap_weights(
    state: dict[str, Any], model
) -> tuple[dict[str, Any], list[str]]:
    """Strip the module's prefix from each weight, reporting what nothing claimed.

    The dropped list is not noise. A typed-decision module stores task heads next to its
    encoder, and those weights are genuinely not part of the encoder byteification
    distils into -- but a load that quietly discarded them would be indistinguishable from
    one that found a checkpoint with the wrong prefix, so they are named.
    """
    remapped: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in state.items():
        for prefix in WEIGHT_PREFIXES:
            if key.startswith(prefix):
                remapped[key[len(prefix) :]] = value
                break
        else:
            dropped.append(key)
    return remapped, dropped


def _tokenizer_fields(tokenizer_file: Path) -> dict[str, Any]:
    """The special tokens a ``tokenizer.json``'s sibling config declares.

    ``AutoTokenizer`` reads these from ``tokenizer_config.json`` next to the vocabulary. A
    module layout keeps both inside the module directory, which is exactly the path
    ``AutoTokenizer`` was never told to look at.
    """
    config_file = tokenizer_file.parent / "tokenizer_config.json"
    if not config_file.exists():
        return {}
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    fields = (
        "unk_token",
        "pad_token",
        "cls_token",
        "sep_token",
        "mask_token",
        "bos_token",
        "eos_token",
        "model_max_length",
        "clean_up_tokenization_spaces",
    )
    return {key: raw[key] for key in fields if raw.get(key) is not None}