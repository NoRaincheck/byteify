"""The fidelity metric itself: what it measures, and what it must not depend on.

Three defects in here were invisible from the outside, because every number the metric
produced was plausible:

* centring was applied **per batch**. Removing the dominant direction means subtracting the
  mean of the population, and a batch of four does not estimate a mean -- the same Stage 1
  checkpoint measured centred cosine 0.885 through this function with batches of sixteen and
  0.672 with batches of four, on identical held-out texts. A gate that scores an artifact by
  how its texts happened to be grouped is not measuring the artifact.
* ``cos_emb_raw`` was computed from the already-centred vectors, so it was *identical to*
  ``cos_emb_c`` by construction. The one number that shows the anisotropy ADR 0005 exists to
  expose was reporting the centred value under a second name.
* training-time patching asked for a fixed number of patches regardless of how many byte
  positions a batch had, and raised on any corpus of short texts.

Everything here runs on the synthetic fixture, offline, in seconds.
"""

from __future__ import annotations

import pytest
import torch

from byteify.fidelity import _budgeted_patch_reps, measure_fidelity
from byteify.inference import ByteifiedEmbedder
from byteify.source import LocalEncoderSpec, build_byteified_encoder
from byteify.stitching import encode_texts


def _batches(texts, tokenizer, size):
    return [
        encode_texts(texts[i : i + size], tokenizer, device="cpu")
        for i in range(0, len(texts), size)
    ]


def _embedder(tiny_source_encoder, tiny_source_tokenizer, budget=8):
    # Seeded, because these assertions are about the shape of the numbers -- whether raw
    # cosine exceeds centred, whether they differ at all -- and an unseeded 32-wide encoder
    # on a synthetic corpus satisfies them only by luck. Order-dependence in a metric test
    # would be the same defect the metric had.
    torch.manual_seed(0)
    encoder = build_byteified_encoder(
        tiny_source_encoder,
        tiny_source_tokenizer,
        LocalEncoderSpec(layers=1, ffn_hidden=32, window=8),
    ).eval()
    return ByteifiedEmbedder(encoder, tiny_source_encoder, n_patches=budget).eval()


def test_the_fidelity_numbers_do_not_depend_on_how_the_texts_were_batched(
    tiny_source_encoder, tiny_source_tokenizer, corpus
):
    """One evaluation set, one number. Batching is an implementation detail of the loop."""
    texts = corpus[:24]
    embedder = _embedder(tiny_source_encoder, tiny_source_tokenizer)

    by_four = measure_fidelity(
        lambda batch: _budgeted_patch_reps(embedder, batch),
        tiny_source_encoder,
        _batches(texts, tiny_source_tokenizer, 4),
    )
    by_twelve = measure_fidelity(
        lambda batch: _budgeted_patch_reps(embedder, batch),
        tiny_source_encoder,
        _batches(texts, tiny_source_tokenizer, 12),
    )

    for metric in ("cos_emb_c", "sim_corr", "boundary_acc"):
        assert by_four[metric] == pytest.approx(by_twelve[metric], abs=1e-6), (
            f"{metric} moved with the batching: {by_four[metric]} at batch 4, "
            f"{by_twelve[metric]} at batch 12"
        )


def test_the_raw_cosine_is_not_the_centred_one_renamed(
    tiny_source_encoder, tiny_source_tokenizer, corpus
):
    """Pooled embeddings are anisotropic; the raw number has to show it.

    `cos_emb_raw` computed from the centred vectors equalled `cos_emb_c` exactly, which
    meant the anisotropy warning this project is built around could never fire.
    """
    texts = corpus[:24]
    embedder = _embedder(tiny_source_encoder, tiny_source_tokenizer)
    metrics = measure_fidelity(
        lambda batch: _budgeted_patch_reps(embedder, batch),
        tiny_source_encoder,
        _batches(texts, tiny_source_tokenizer, 4),
    )

    assert metrics["cos_emb_raw"] != metrics["cos_emb_c"], (
        "the raw and centred cosines are identical, so one of them is not being computed "
        "from uncentred embeddings"
    )
    # Raw cosine of pooled embeddings sits high whatever the content; that is the point.
    assert metrics["cos_emb_raw"] > metrics["cos_emb_c"]
    # And the dominant direction is what it agrees on, which is what makes raw 0.99 a
    # number that should never be quoted alone.
    assert -1.0 <= metrics["dominant_dir_cosine"] <= 1.0


def test_a_batch_shorter_than_the_patch_axis_is_filled_rather_than_refused(
    tiny_source_encoder, tiny_source_tokenizer
):
    """The serving path clamps; the training path did not, and raised on short texts.

    `n_patches` is a property of the exported graph's shape, not of a batch. Asking for
    more patches than the batch has byte positions cannot be satisfied and used to raise,
    which killed the contrastive stage on any corpus of short documents.
    """
    embedder = _embedder(tiny_source_encoder, tiny_source_tokenizer, budget=64)
    short = ["byte", "patch", "gate", "boundary"]
    rows = encode_texts(short, tiny_source_tokenizer, device="cpu")

    assert rows.byte_mask.shape[1] < embedder.n_patches, "fixture is not exercising the clamp"
    reps, mask, _probs = _budgeted_patch_reps(embedder, rows)

    assert reps.shape[0] == len(short)
    assert reps.shape[1] <= rows.byte_mask.shape[1]
    assert mask.shape == (len(short), reps.shape[1])
    assert torch.isfinite(reps).all()


def test_encode_moves_host_built_inputs_onto_the_embedder_device(tiny_source_encoder,
                                                                 tiny_source_tokenizer,
                                                                 corpus):
    """`byte_ids_for_inference` builds tensors from Python integers, on the host.

    On any device other than CPU the first op was handed a CPU tensor and the call died with
    an opaque device mismatch -- which is how the retrieval gate failed the first time it ran.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("no non-CPU device available")

    # A copy: `tiny_source_encoder` is session-scoped, and moving it to MPS in place would
    # leave every later CPU test in the suite holding a module on the wrong device.
    import copy

    source = copy.deepcopy(tiny_source_encoder)
    embedder = _embedder(source, tiny_source_tokenizer).to("mps")
    source.to("mps")

    embeddings = embedder.encode(tiny_source_tokenizer, corpus[:3], max_bytes=128)

    assert embeddings.device.type == "mps"
    assert torch.isfinite(embeddings).all()


def pytest_approx(value, tolerance):
    import pytest

    return pytest.approx(value, abs=tolerance)