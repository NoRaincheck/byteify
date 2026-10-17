"""The manifest is a byteified artifact's only entry point (Ticket #13).

The bar for these tests is what a user of the artifact can observe: a manifest field, a
refusal, an embedding. The one that matters most is the round trip -- save an artifact,
load it through its manifest alone with nothing else supplied, and get the same embeddings
back. An artifact that loads but computes something else is useless, and that is the class
of defect this whole ticket exists to close.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from byteify import ByteifiedEmbedder, Manifest, export_artifact, load_artifact
from byteify.bytestream import rule_for_family
from byteify.manifest import (
    ManifestError,
    UnsupportedSchemaVersion,
    manifest_from_checkpoint,
    read_trie,
    write_trie,
)
from byteify.source import LocalEncoderSpec, build_byteified_encoder

# -- fixtures ------------------------------------------------------------


@pytest.fixture()
def trained_encoder(tmp_path: Path, tiny_source_encoder, tiny_source_tokenizer):
    """A byteified encoder with a checkpoint on disk, trained for a handful of steps.

    Deliberately *not* the default architecture: the regression this ticket exists for is
    an export path that hardcodes its configuration, which only shows up on a checkpoint
    that was trained with something else.
    """
    spec = LocalEncoderSpec(layers=2, ffn_hidden=64, window=16, boundary_predictor="mlp")
    encoder = build_byteified_encoder(tiny_source_encoder, tiny_source_tokenizer, spec)
    path = tmp_path / "stage1.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "config": {"seed": 3, "model": "tiny", **spec.to_config()},
            "step": 7,
        },
        path,
    )
    return encoder, path


@pytest.fixture()
def rule(tiny_source_tokenizer):
    from byteify.alignment import discover_special_bytes

    specials = discover_special_bytes(tiny_source_tokenizer)
    return rule_for_family("bytelevel", n_byte_values=256 + len(specials), special_tokens=specials)


# -- the manifest is enough ------------------------------------------------


def test_an_artifact_loads_through_its_manifest_alone(
    tmp_path: Path, trained_encoder, tiny_source_encoder, tiny_source_tokenizer, rule
):
    """Export an artifact, reload it with only the manifest, and embed text.

    Nothing else is supplied: no checkpoint path, no tokenizer, no architecture flags.
    """
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()

    artifact = tmp_path / "artifact"
    export_artifact(
        embedder,
        tiny_source_tokenizer,
        checkpoint,
        artifact,
        budget=16,
        max_bytes=128,
    )

    _, reloaded = load_artifact(artifact)

    texts = ["the quick brown fox", "def f(x): return x", "日本語のテキスト"]
    for text in texts:
        ids, expanded, mask = reloaded.inputs_for_text(text)
        with torch.no_grad():
            original = embedder(ids, expanded, mask)
            through_manifest = reloaded(ids, expanded, mask)
        assert torch.allclose(original, through_manifest, atol=1e-6)


def test_the_manifest_carries_the_encoder_architecture(
    tmp_path: Path, trained_encoder, tiny_source_encoder, tiny_source_tokenizer, rule
):
    """A checkpoint trained with a non-default interface reloads and describes itself."""
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()

    manifest = export_artifact(
        embedder,
        tiny_source_tokenizer,
        checkpoint,
        tmp_path / "artifact",
        budget=16,
        max_bytes=128,
    )

    assert manifest.encoder_arch["layers"] == 2
    assert manifest.encoder_arch["ffn_hidden"] == 64
    assert manifest.encoder_arch["window"] == 16
    # And it reads back as a spec that rebuilds the same interface.
    spec = LocalEncoderSpec.from_config(manifest.encoder_arch)
    assert (spec.layers, spec.ffn_hidden, spec.window) == (2, 64, 16)


def test_the_manifest_records_the_byte_value_count_the_weights_actually_have(
    trained_encoder, tiny_source_encoder, rule
):
    """The byte table's size comes from the weights, not from a config key.

    A 263-row byte table described as 256 is not cosmetic: a loader that trusts the
    manifest builds a byte embedding of the wrong shape and fails at load time.
    """
    encoder, checkpoint = trained_encoder
    real_rows = int(encoder.embed.byte.weight.shape[0])
    assert real_rows > 256  # the fixture must actually exercise this

    manifest = manifest_from_checkpoint(
        checkpoint,
        source=tiny_source_encoder,
        budget=16,
        max_bytes=128,
        rule=rule,
    )
    assert manifest.n_byte_values == real_rows
    assert manifest.vocab_size == int(encoder.embed.suffix.weight.shape[0])


def test_the_patch_budget_is_an_argument_not_a_default(
    trained_encoder, tiny_source_encoder, rule
):
    """The budget the manifest states is the budget the artifact was built with.

    The budget is the axis the serving guarantee rests on (ADR 0002). A manifest that
    defaulted it would describe an artifact nobody built.
    """
    _, checkpoint = trained_encoder
    for budget in (16, 64, 200):
        manifest = manifest_from_checkpoint(
            checkpoint,
            source=tiny_source_encoder,
            budget=budget,
            max_bytes=128,
            rule=rule,
        )
        assert manifest.budget == budget


def test_a_manifest_without_a_budget_is_refused(tmp_path: Path, rule):
    with pytest.raises(TypeError):
        # There is deliberately no default for budget or max_bytes.
        manifest_from_checkpoint(tmp_path / "nope.pt", source=object(), rule=rule)


def test_the_manifest_records_the_family_the_trie_rule_and_coverage(
    tmp_path: Path, trained_encoder, tiny_source_encoder, tiny_source_tokenizer, rule
):
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()
    manifest = export_artifact(
        embedder,
        tiny_source_tokenizer,
        checkpoint,
        tmp_path / "artifact",
        budget=16,
        max_bytes=128,
        coverage_texts=["the quick brown fox", "日本語のテキスト"],
    )
    assert manifest.family == "bytelevel"
    assert manifest.byte_stream_rule == "nfc"
    assert manifest.trie.coverage is not None
    assert manifest.trie.coverage_bytes > 0
    assert manifest.trie.sha256


def test_inference_inputs_need_no_source_tokenizer(
    tmp_path: Path, trained_encoder, tiny_source_encoder, tiny_source_tokenizer, rule
):
    """The shipped trie is enough to expand bytes into suffix ids."""
    from byteify.alignment import align_text

    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()
    artifact = tmp_path / "artifact"
    export_artifact(
        embedder, tiny_source_tokenizer, checkpoint, artifact,
        budget=16, max_bytes=128,
    )
    _, reloaded = load_artifact(artifact)

    text = "日本語のテキスト"
    ids, expanded, _ = reloaded.inputs_for_text(text)
    # Same bytes training would have produced, and the same suffix ids the trie yields.
    from byteify.alignment import discover_special_bytes

    training = align_text(
        text,
        tiny_source_tokenizer,
        special_token_ids=discover_special_bytes(tiny_source_tokenizer),
        add_special_tokens=True,
    )
    assert ids[0].tolist() == list(training.byte_ids)
    assert len(expanded[0]) == len(training.byte_ids)


def test_a_tampered_trie_is_refused(tmp_path: Path, trained_encoder, tiny_source_encoder,
                                    tiny_source_tokenizer, rule):
    """The manifest records the trie's hash, so a corrupted artifact does not load."""
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()
    artifact = tmp_path / "artifact"
    manifest = export_artifact(
        embedder, tiny_source_tokenizer, checkpoint, artifact,
        budget=16, max_bytes=128,
    )

    trie_path = artifact / manifest.trie.path
    payload = json.loads(trie_path.read_text())
    payload[min(payload)] = "00"
    trie_path.write_text(json.dumps(payload))

    with pytest.raises(ManifestError, match="sha256"):
        load_artifact(artifact)


def test_a_missing_trie_is_refused(tmp_path: Path, trained_encoder, tiny_source_encoder,
                                   tiny_source_tokenizer, rule):
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()
    artifact = tmp_path / "artifact"
    export_artifact(
        embedder, tiny_source_tokenizer, checkpoint, artifact,
        budget=16, max_bytes=128,
    )
    (artifact / "trie.json").unlink()
    with pytest.raises(ManifestError, match="trie"):
        load_artifact(artifact)


# -- versioning -----------------------------------------------------------


def test_an_unrecognised_major_version_is_refused(valid_manifest: Manifest):
    from dataclasses import replace

    future = replace(valid_manifest, schema_version="99.0")
    with pytest.raises(UnsupportedSchemaVersion):
        Manifest.from_json(future.to_json())


def test_a_later_minor_version_still_loads(valid_manifest: Manifest):
    from dataclasses import replace

    newer_minor = replace(valid_manifest, schema_version="1.7")
    assert Manifest.from_json(newer_minor.to_json()).budget == valid_manifest.budget


def test_a_malformed_version_is_refused(valid_manifest: Manifest):
    from dataclasses import replace

    broken = replace(valid_manifest, schema_version="one")
    with pytest.raises(ManifestError):
        Manifest.from_json(broken.to_json())


def test_a_refused_version_is_refused_on_disk_too(valid_manifest: Manifest, tmp_path: Path):
    from dataclasses import replace

    path = tmp_path / "manifest.json"
    replace(valid_manifest, schema_version="2.0").write_to(path)
    with pytest.raises(UnsupportedSchemaVersion):
        Manifest.from_file(path)


# -- validation -----------------------------------------------------------


def test_a_valid_manifest_has_nothing_to_report(valid_manifest: Manifest):
    assert valid_manifest.validate() == []


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("source_id", "source_id is empty"),
        ("family", "family is empty"),
        ("byte_stream_rule", "byte_stream_rule is empty"),
        ("source_d_model", "source_d_model must be > 0"),
        ("source_n_layers", "source_n_layers must be > 0"),
        ("budget", "budget must be > 0"),
        ("max_bytes", "max_bytes must be > 0"),
        ("vocab_size", "vocab_size must be > 0"),
        ("byte_stream", "byte_stream is empty"),
    ],
)
def test_each_missing_field_is_named(valid_manifest: Manifest, field: str, expected: str):
    from dataclasses import replace

    bad = replace(valid_manifest, **{field: type(getattr(valid_manifest, field))()})
    assert any(expected in e for e in bad.validate()), bad.validate()


def test_a_byte_table_smaller_than_the_byte_alphabet_is_refused(valid_manifest: Manifest):
    from dataclasses import replace

    bad = replace(valid_manifest, n_byte_values=255)
    assert any("n_byte_values" in e for e in bad.validate())


def test_an_encoder_arch_missing_a_field_is_refused(valid_manifest: Manifest):
    from dataclasses import replace

    arch = {k: v for k, v in valid_manifest.encoder_arch.items() if k != "ffn_hidden"}
    bad = replace(valid_manifest, encoder_arch=arch)
    assert any("ffn_hidden" in e for e in bad.validate())


def test_an_artifact_whose_recorded_gate_failed_is_refused(valid_manifest: Manifest):
    from dataclasses import replace

    from byteify import GateResults

    failed = replace(valid_manifest, gate=GateResults(cos_emb=0.1, gate_passed=False))
    assert any("gate" in e for e in failed.validate())


def test_an_invalid_manifest_on_disk_is_refused_with_its_reasons(
    valid_manifest: Manifest, tmp_path: Path
):
    from dataclasses import replace

    path = tmp_path / "manifest.json"
    replace(valid_manifest, budget=0, source_id="").write_to(path)
    with pytest.raises(ManifestError) as excinfo:
        Manifest.from_file(path)
    assert "budget" in str(excinfo.value)


def test_loading_a_missing_manifest_says_so(tmp_path: Path):
    with pytest.raises(ManifestError, match="no manifest"):
        Manifest.load(tmp_path)


def test_a_manifest_that_is_not_json_says_so(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.from_file(path)


# -- round trip -----------------------------------------------------------


def test_a_manifest_survives_json_unchanged(valid_manifest: Manifest):
    from dataclasses import asdict

    assert asdict(Manifest.from_json(valid_manifest.to_json())) == asdict(valid_manifest)


def test_the_gate_survives_json_with_its_thresholds(valid_manifest: Manifest):
    from dataclasses import replace

    from byteify import GateResults

    gated = replace(
        valid_manifest,
        gate=GateResults(cos_emb=0.93, sim_corr=0.88, thresholds={"cos_emb": 0.90}),
    )
    back = Manifest.from_json(gated.to_json())
    assert back.gate.thresholds == {"cos_emb": 0.90}
    assert back.gate.cos_emb == 0.93


def test_writing_a_manifest_is_atomic(valid_manifest: Manifest, tmp_path: Path):
    """A half-written manifest is worse than none: it names an artifact that may not exist."""
    path = tmp_path / "manifest.json"
    valid_manifest.write_to(path)
    assert json.loads(path.read_text())["budget"] == 64
    assert not list(tmp_path.glob("*.tmp"))


# -- the trie -------------------------------------------------------------


def test_the_trie_survives_a_write_and_read(tmp_path: Path):
    token_bytes = {0: b"a", 1: b"the", 2: b"\xf0\x9f\x99\x82", 3: b""}
    path = tmp_path / "trie.json"
    digest = write_trie(path, token_bytes)
    assert len(digest) == 64
    # The empty entry is dropped: a zero-length token covers no byte.
    assert read_trie(path) == {0: b"a", 1: b"the", 2: b"\xf0\x9f\x99\x82"}


def test_an_artifact_written_atomically_leaves_no_temp_files(
    tmp_path: Path, trained_encoder, tiny_source_encoder, tiny_source_tokenizer, rule
):
    encoder, checkpoint = trained_encoder
    embedder = ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=16).eval()
    artifact = tmp_path / "artifact"
    export_artifact(
        embedder, tiny_source_tokenizer, checkpoint, artifact,
        budget=16, max_bytes=128,
    )
    assert not list(artifact.glob("*.tmp"))
    assert (artifact / "manifest.json").exists()