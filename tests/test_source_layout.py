"""A source model that is not a plain HuggingFace checkpoint.

Most published embedding models are a plain checkpoint: ``config.json``, weights and a
tokenizer at the repository root, which ``AutoModel`` reads without help. Some are not.
``hotchpotch/bekko-system-one-v0-17m`` is one: its encoder lives in a module directory
named by ``modules.json``, is configured by ``backbone_config.json``, and its weights carry
the module's own prefix with three task heads stored beside the encoder.

That layout matters more than it looks. A byteified encoder delegates its patch
representations to the frozen source encoder -- level 1's fidelity guarantee *is* that
delegation -- so the source model is re-fetched every time an artifact is loaded. A
repository byteify can build from but cannot serve is not a supported target, and the
module layout is the difference between the two.

These tests build the layout on disk from the synthetic model, so they run offline in
seconds and fail the same way whether the defect is in the detection, the remapping or the
refusal.
"""

from __future__ import annotations

import json

import pytest
import torch


def _module_layout(tmp_path, model, tokenizer, *, prefix="encoder.backbone.", heads=True):
    """Write a source model out in the custom-code module layout, as the Hub holds it."""
    root = tmp_path / "module-repo"
    module = root / "0_Module"
    (module / "tokenizer").mkdir(parents=True)

    (root / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Module", "type": "mod.Module"}]),
        encoding="utf-8",
    )
    model.config.save_pretrained(module)          # writes config.json
    (module / "config.json").rename(module / "backbone_config.json")

    state = {f"{prefix}{name}": tensor for name, tensor in model.state_dict().items()}
    if heads:
        # A typed-decision module keeps its heads next to the encoder. They are not
        # encoder weights, and a load that discarded them silently would look exactly
        # like one that found the wrong prefix.
        for head in ("choice", "noul", "score"):
            state[f"heads.{head}.weight"] = torch.zeros(1, model.config.hidden_size)
            state[f"heads.{head}.bias"] = torch.zeros(1)
    from safetensors.torch import save_file

    save_file(state, str(module / "model.safetensors"))

    tokenizer.save_pretrained(module / "tokenizer")
    return root


def test_a_module_layout_resolves_to_the_same_weights_as_a_plain_checkpoint(
    tmp_path, tiny_source_dir, tiny_tokenizers
):
    """The whole point: the two layouts must produce the same encoder, not merely load."""
    from transformers import AutoModel

    from byteify.hub import resolve_source_model

    plain = AutoModel.from_pretrained(tiny_source_dir["modernbert"])
    root = _module_layout(tmp_path, plain, tiny_tokenizers["bytelevel"])

    resolved = resolve_source_model(str(root))

    assert resolved.layout == "module"
    assert resolved.module_path == "0_Module"
    for (name, expected), (_, got) in zip(
        plain.state_dict().items(), resolved.model.state_dict().items()
    ):
        assert torch.equal(expected, got), f"{name} differs between the two layouts"


def test_the_heads_beside_the_encoder_are_reported_not_silently_dropped(
    tmp_path, tiny_source_dir, tiny_tokenizers
):
    from transformers import AutoModel

    from byteify.hub import resolve_source_model

    root = _module_layout(tmp_path, AutoModel.from_pretrained(tiny_source_dir["modernbert"]),
                          tiny_tokenizers["bytelevel"])
    resolved = resolve_source_model(str(root))

    assert sorted(resolved.dropped) == [
        "heads.choice.bias", "heads.choice.weight",
        "heads.noul.bias", "heads.noul.weight",
        "heads.score.bias", "heads.score.weight",
    ]


def test_the_module_tokenizer_resolves_through_the_same_route(tmp_path, tiny_source_dir,
                                                             tiny_tokenizers):
    """A build that read the weights one way and the vocabulary another would disagree."""
    from transformers import AutoModel

    from byteify.hub import resolve_source_tokenizer
    from byteify.vocab import resolve_token_bytes

    root = _module_layout(tmp_path, AutoModel.from_pretrained(tiny_source_dir["modernbert"]),
                          tiny_tokenizers["bytelevel"])
    tokenizer = resolve_source_tokenizer(str(root))

    # Same vocabulary as the one the module was written with: that is the assertion. If
    # the loader silently produced an empty or differently-segmented tokenizer, the
    # byte stream and the boundary supervision would both be wrong.
    assert resolve_token_bytes(tokenizer).token_bytes == resolve_token_bytes(
        tiny_tokenizers["bytelevel"]
    ).token_bytes


def test_a_plain_checkpoint_still_goes_through_auto_model(tiny_source_dir):
    """The module layout is a fallback. The ordinary path must be untouched by it."""
    from byteify.hub import resolve_source_model

    resolved = resolve_source_model(str(tiny_source_dir["modernbert"]))
    assert resolved.layout == "checkpoint"
    assert resolved.dropped == ()


def test_weights_that_do_not_match_the_config_are_refused_not_partially_loaded(tmp_path):
    """Loading the overlap would leave half the encoder random and nothing would say so."""
    from safetensors.torch import save_file
    from transformers import ModernBertConfig

    root = tmp_path / "broken"
    module = root / "0_Module"
    module.mkdir(parents=True)
    (root / "modules.json").write_text(json.dumps([{"path": "0_Module"}]), encoding="utf-8")
    ModernBertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, vocab_size=32, pad_token_id=0,
                     cls_token_id=1, sep_token_id=2).save_pretrained(module)
    (module / "config.json").rename(module / "backbone_config.json")
    # One encoder tensor, of the right family but far too few of them.
    save_file(
        {"encoder.backbone.final_norm.weight": torch.zeros(16)},
        str(module / "model.safetensors"),
    )

    from byteify.hub import resolve_source_model

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_source_model(str(root))
    assert "does not match" in str(excinfo.value)


def test_a_repository_that_is_neither_layout_says_so(tmp_path):
    """Two failures, two reasons, named -- not a traceback from inside a library."""
    from byteify.hub import resolve_source_model

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_source_model(str(empty))
    message = str(excinfo.value)
    assert "not a plain checkpoint" in message
    assert "no encoder config" in message


def test_a_module_layout_builds_and_serves_through_the_manifest_path(
    tmp_path, tiny_source_dir, tiny_tokenizers
):
    """End to end: the module layout resolves at build time and again at load time.

    The second resolution is the one that matters. An artifact's manifest names its source
    by id, and `byteify encode` re-fetches that source on every call; a layout that
    resolved only while training would produce an artifact that cannot be served.
    """
    from transformers import AutoModel

    from byteify.hub import resolve_source_tokenizer
    from byteify.source import resolve_source_encoder
    from byteify.stitching import encode_texts
    from byteify.vocab import build_suffix_trie, resolve_token_bytes

    root = _module_layout(tmp_path, AutoModel.from_pretrained(tiny_source_dir["modernbert"]),
                          tiny_tokenizers["bytelevel"])
    source = resolve_source_encoder(str(root))
    tokenizer = resolve_source_tokenizer(str(root))

    assert source.family == "modernbert"
    assert source.layout == "module"
    assert source.model_id == str(root)

    # The three things byteification needs from a source model, through the module layout.
    assert source.embedding_matrix().shape[1] == source.d_model
    rows = encode_texts(["a byte native stream", "another one"], tokenizer, device="cpu")
    with torch.no_grad():
        hidden = source.forward_layers(
            source.embedding_matrix()[rows.input_ids], rows.attention_mask, n_layers=1
        )
    assert hidden.shape[-1] == source.d_model
    assert torch.isfinite(hidden).all()
    assert resolve_token_bytes(tokenizer).strategy == "bytelevel"
    assert len(build_suffix_trie(tokenizer)) > 0


def test_a_hub_id_resolves_through_the_cache_with_a_narrow_allowlist(
    tmp_path, tiny_source_dir, tiny_tokenizers, monkeypatch
):
    """A hub id becomes a directory through the cache, and only the files byteify needs.

    The allowlist is part of the contract rather than an optimisation: an artifact's
    manifest names its source, and `byteify encode` re-fetches that source on every call.
    If resolving it pulled the repository's ONNX exports and evaluation scripts, every
    serving call would cost tens of megabytes of download to compute a 256-wide embedding.
    """
    from transformers import AutoModel

    from byteify.hub import resolve_source_model, resolve_source_tokenizer

    root = _module_layout(tmp_path, AutoModel.from_pretrained(tiny_source_dir["modernbert"]),
                          tiny_tokenizers["bytelevel"])
    requested: list[list[str]] = []

    def cached_download(repo_id, *, allow_patterns=None, **kwargs):
        requested.append(list(allow_patterns or []))
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", cached_download)

    model = resolve_source_model("synthetic/module-repo")
    tokenizer = resolve_source_tokenizer("synthetic/module-repo")

    assert model.layout == "module"
    assert tokenizer.vocab_size > 0
    assert requested, "the repository was never resolved through the cache"
    for patterns in requested:
        assert "**/backbone_config.json" in patterns
        assert "**/tokenizer/tokenizer.json" in patterns
        assert not any(pattern.endswith(".onnx") for pattern in patterns)