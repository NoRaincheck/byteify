"""Post-training a byteified model with task arithmetic, at no extra training cost.

The paper's Fig. 3 result: a byteified model can absorb post-training from an existing
subword checkpoint by adding the *weight difference* of the post-trained model relative
to its base, for the transformer layers only::

    theta_merged = theta_byteified + alpha * (theta_instruct - theta_base)

This works because Stage 1 leaves the global model bit-identical to the source model --
byteification replaces the input and output interface, not the transformer body. So the
delta an instruction-tuned checkpoint contributes is expressed in the same coordinate
system as the byteified model's transformer layers, and needs no retraining.

One subtlety the paper calls out and we preserve: **embeddings are excluded**. The
post-trained checkpoint's input embeddings are indexed by the subword vocabulary and
have no counterpart on the byte axis, and resetting them to the base model's values is
what makes the merge safe. Only parameters that exist in all three models, inside the
transformer body, are touched.

Parameter names in a byteified checkpoint are prefixed by the wrapper (``source.model...``),
so keys are matched on their **suffix** against the plain HuggingFace state dict.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping

import torch

# Parameters that must not be task-arithmetic'd: the vocabulary-indexed embedding,
# its norm, and any head. The byteified model indexes embeddings by byte, so a subword
# embedding delta has nowhere to land.
_DEFAULT_EXCLUDE = (
    "embeddings",
    "tok_embeddings",
    "word_embeddings",
    "embed_positions",
    "position_embeddings",
    "lm_head",
    "head.",
    "final_norm",
    "LayerNorm",
    "norm.",
)


def _matches(key: str, patterns: Iterable[str]) -> bool:
    return any(p in key for p in patterns)


def delta(
    instruct_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """``theta_instruct - theta_base`` over the keys present in both."""
    out = {}
    for k, v in instruct_state.items():
        if k in base_state and base_state[k].shape == v.shape:
            out[k] = v.float() - base_state[k].float()
    return out


def apply_delta(
    state: MutableMapping[str, torch.Tensor],
    task_delta: Mapping[str, torch.Tensor],
    *,
    alpha: float = 1.0,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = _DEFAULT_EXCLUDE,
    key_map: Mapping[str, str] | None = None,
) -> list[str]:
    """Add ``alpha * task_delta`` into ``state`` in place. Returns the keys changed."""
    changed: list[str] = []
    for src_key, d in task_delta.items():
        if include is not None and not _matches(src_key, include):
            continue
        if _matches(src_key, exclude):
            continue
        dst_key = key_map.get(src_key, src_key) if key_map else src_key
        if dst_key not in state:
            continue
        target = state[dst_key]
        if target.shape != d.shape:
            raise ValueError(f"shape mismatch for {dst_key}: {tuple(target.shape)} vs {tuple(d.shape)}")
        state[dst_key] = target + alpha * d.to(target.dtype)
        changed.append(dst_key)
    return changed


def suffix_key_map(
    wrapped_state: Mapping[str, torch.Tensor],
    plain_state: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Map plain HuggingFace parameter names onto their wrapped counterparts.

    Byteified checkpoints nest the source model under a wrapper prefix (``source.``), so
    exact-key lookup fails. Matching on the tail of the name is robust as long as the
    tail is unique, which it is for transformer weights.
    """
    mapping: dict[str, str] = {}
    for plain in plain_state:
        matches = [w for w in wrapped_state if w == plain or w.endswith("." + plain)]
        if len(matches) == 1:
            mapping[plain] = matches[0]
    return mapping


def task_arithmetic_merge(
    wrapped_state: MutableMapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    instruct_state: Mapping[str, torch.Tensor],
    *,
    alpha: float = 1.0,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = _DEFAULT_EXCLUDE,
) -> dict[str, object]:
    """Merge a post-trained checkpoint's transformer delta into a byteified checkpoint.

    Returns a report: which keys changed, how many deltas were skipped and why.
    """
    task_delta = delta(instruct_state, base_state)
    key_map = suffix_key_map(wrapped_state, instruct_state)

    skipped_unmapped = [k for k in task_delta if k not in key_map]
    before = {k: wrapped_state[k].clone() for k in wrapped_state if k in key_map}
    changed = apply_delta(
        wrapped_state,
        task_delta,
        alpha=alpha,
        include=include,
        exclude=exclude,
        key_map=key_map,
    )
    skipped_excluded = [k for k in task_delta if k not in changed and k in key_map]
    return {
        "alpha": alpha,
        "n_delta": len(task_delta),
        "n_changed": len(changed),
        "changed": changed,
        "skipped_unmapped": skipped_unmapped,
        "skipped_excluded": skipped_excluded,
        "n_before": before,
    }


def describe_transformer_scope(model_id: str) -> re.Pattern[str]:
    """Matcher for "the transformer body", used as an ``include`` filter."""
    return re.compile(r"(layers|layer)\.\d+\.")


def save_merged(
    wrapped_state: Mapping[str, torch.Tensor],
    out_path: str,
    meta: dict[str, object],
) -> None:
    torch.save({"state_dict": dict(wrapped_state), "meta": {k: v for k, v in meta.items() if k != "n_before"}}, out_path)