"""Task arithmetic: post-train a byteified model from an existing subword checkpoint."""

import pytest
import torch

from byteify.task_arithmetic import (
    apply_delta,
    delta,
    suffix_key_map,
    task_arithmetic_merge,
)

TRANSFORMER_KEYS = (
    "layers.0.attn.Wqkv.weight",
    "layers.0.mlp.Wi.weight",
    "layers.1.attn.Wqkv.weight",
)


def make_state(seed: int, prefix: str = "") -> dict[str, torch.Tensor]:
    """An independent weight set. Seeds must differ or every delta is trivially zero."""
    gen = torch.Generator().manual_seed(seed)
    return {
        f"{prefix}embeddings.tok_embeddings.weight": torch.randn(10, 4, generator=gen),
        f"{prefix}layers.0.attn.Wqkv.weight": torch.randn(4, 4, generator=gen),
        f"{prefix}layers.0.mlp.Wi.weight": torch.randn(4, 4, generator=gen),
        f"{prefix}layers.1.attn.Wqkv.weight": torch.randn(4, 4, generator=gen),
        f"{prefix}final_norm.weight": torch.randn(4, generator=gen),
    }


def test_delta_is_instruct_minus_base():
    base = make_state(1)
    instruct = make_state(2)
    d = delta(instruct, base)
    assert set(d) == set(base)
    for k in base:
        assert torch.allclose(d[k], instruct[k] - base[k])
    assert any(d[k].abs().sum() > 0 for k in d), "deltas must be non-zero"


def test_suffix_key_map_resolves_wrapper_prefix():
    plain = make_state(1)
    wrapped = make_state(1, "source.model.")
    mapping = suffix_key_map(wrapped, plain)
    assert mapping["layers.0.mlp.Wi.weight"] == "source.model.layers.0.mlp.Wi.weight"


def test_merge_adds_alpha_times_delta_to_transformer_layers():
    base, instruct = make_state(1), make_state(2)
    wrapped = make_state(3, "source.model.")
    original = {k: v.clone() for k, v in wrapped.items()}

    report = task_arithmetic_merge(wrapped, base, instruct, alpha=1.0)
    assert report["n_changed"] == len(TRANSFORMER_KEYS)

    for key in TRANSFORMER_KEYS:
        expected = original[f"source.model.{key}"] + (instruct[key] - base[key])
        assert torch.allclose(wrapped[f"source.model.{key}"], expected), key


def test_merge_scales_linearly_with_alpha():
    base, instruct = make_state(1), make_state(2)
    a, b = make_state(3, "source.model."), make_state(3, "source.model.")
    task_arithmetic_merge(a, base, instruct, alpha=0.5)
    task_arithmetic_merge(b, base, instruct, alpha=1.0)

    key = "source.model.layers.0.attn.Wqkv.weight"
    shift_half = (instruct["layers.0.attn.Wqkv.weight"] - base["layers.0.attn.Wqkv.weight"])
    assert torch.allclose(b[key] - a[key], 0.5 * shift_half, atol=1e-6)


def test_merge_never_touches_embeddings_or_final_norm():
    """The paper's subtlety: embeddings must stay resettable to the base values."""
    base, instruct = make_state(1), make_state(2)
    wrapped = make_state(3, "source.model.")
    before = {k: v.clone() for k, v in wrapped.items()}

    report = task_arithmetic_merge(wrapped, base, instruct)

    for key in ("source.model.embeddings.tok_embeddings.weight", "source.model.final_norm.weight"):
        assert torch.equal(wrapped[key], before[key]), key
    excluded = set(report["skipped_excluded"])
    assert "embeddings.tok_embeddings.weight" in excluded
    assert "final_norm.weight" in excluded


def test_zero_alpha_is_a_no_op():
    base, instruct = make_state(1), make_state(2)
    wrapped = make_state(3, "source.model.")
    before = {k: v.clone() for k, v in wrapped.items()}
    task_arithmetic_merge(wrapped, base, instruct, alpha=0.0)
    for k, v in before.items():
        assert torch.equal(wrapped[k], v)


def test_merge_with_identical_instruct_and_base_is_a_no_op():
    base = make_state(1)
    wrapped = make_state(3, "source.model.")
    before = {k: v.clone() for k, v in wrapped.items()}
    task_arithmetic_merge(wrapped, base, make_state(1), alpha=1.0)
    for k, v in before.items():
        assert torch.equal(wrapped[k], v)


def test_apply_delta_rejects_shape_mismatch():
    state = {"layers.0.w": torch.zeros(4, 4)}
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_delta(state, {"layers.0.w": torch.zeros(3, 3)})


def test_merge_reports_keys_present_in_delta_but_absent_from_byteified_model():
    base, instruct = make_state(1), make_state(2)
    base["layers.9.mlp.Wi.weight"] = torch.randn(4, 4)
    instruct["layers.9.mlp.Wi.weight"] = torch.randn(4, 4)
    wrapped = make_state(3, "source.model.")
    report = task_arithmetic_merge(wrapped, base, instruct)
    assert list(report["skipped_unmapped"]) == ["layers.9.mlp.Wi.weight"]


def test_include_filter_restricts_scope_to_named_layers():
    base, instruct = make_state(1), make_state(2)
    wrapped = make_state(3, "source.model.")
    report = task_arithmetic_merge(
        wrapped, base, instruct, include=("layers.1.",), exclude=()
    )
    assert report["n_changed"] == 1
    assert list(report["changed"]) == ["source.model.layers.1.attn.Wqkv.weight"]


def test_end_to_end_merge_shifts_representations_toward_the_instruct_model():
    """The merge must move the model, not merely satisfy a shape check.

    Simulates post-training as a structured perturbation of the transformer
    projections, then checks the merged model's representations end up closer to the
    "post-trained" model than the untouched base is.
    """
    pytest.importorskip("transformers")
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    base_model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-17m")
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    gen = torch.Generator().manual_seed(7)
    instruct_state = {k: v.clone() for k, v in base_state.items()}
    for k, v in instruct_state.items():
        if "layers.0." in k or "layers.3." in k:
            instruct_state[k] = v + 0.02 * torch.randn(v.shape, generator=gen)

    wrapped = {f"source.model.{k}": v.clone() for k, v in base_state.items()}
    report = task_arithmetic_merge(wrapped, base_state, instruct_state, alpha=1.0)
    assert report["n_changed"] > 0

    ids = torch.tensor([[50281, 12092, 64, 14947, 2, 50282]])
    mask = torch.ones_like(ids)
    emb = base_model.embeddings(input_ids=ids)

    def run(state):
        m = AutoModel.from_config(base_model.config)
        m.load_state_dict(state)
        m.eval()
        return HuggingFaceSourceEncoder(m).forward_layers(emb, mask, n_layers=7)

    with torch.no_grad():
        base_h = run(base_state)
        instruct_h = run(instruct_state)
        merged_h = run({k[len("source.model.") :]: v for k, v in wrapped.items()})

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    d_base = rel(base_h, instruct_h)
    d_merged = rel(merged_h, instruct_h)
    assert d_merged < d_base, f"merged {d_merged:.5f} should be below base {d_base:.5f}"