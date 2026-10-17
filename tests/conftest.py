"""Shared fixtures.

Nothing in here may touch the network. Any test needing a source model builds the synthetic
one from :mod:`tests.synthetic` and loads it from a temporary directory, so the suite runs
offline and in seconds. The real anchor model stays a separate, slower tier.
"""

from __future__ import annotations

import pytest

from tests.synthetic import (
    SYNTHETIC_FAMILIES,
    SYNTHETIC_MODEL_TOKENIZER,
    SYNTHETIC_TOKENIZER_FAMILIES,
    build_source_model,
    build_tokenizer,
    synthetic_corpus,
)


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    """Deterministic training text for the tiny tokenizers."""
    return synthetic_corpus()


@pytest.fixture(scope="session")
def tiny_tokenizers(corpus) -> dict[str, object]:
    """One real fast tokenizer per tokenizer family, trained once and reused."""
    return {
        family: build_tokenizer(family, corpus) for family in SYNTHETIC_TOKENIZER_FAMILIES
    }


@pytest.fixture(scope="session")
def tiny_source_dir(tmp_path_factory, tiny_tokenizers):
    """A synthetic source model persisted to disk, one directory per architecture family."""
    root = tmp_path_factory.mktemp("tiny-source")
    paths = {}
    for family in SYNTHETIC_FAMILIES:
        path = root / family
        tokenizer = tiny_tokenizers[SYNTHETIC_MODEL_TOKENIZER[family]]
        build_source_model(family, tokenizer).save_pretrained(path)
        paths[family] = path
    return paths


@pytest.fixture(scope="session")
def tiny_source(tiny_source_dir):
    """The synthetic ModernBERT source model, matching the anchor model's architecture."""
    return tiny_source_dir["modernbert"]


@pytest.fixture(scope="session")
def tiny_source_encoder(tiny_source):
    """The synthetic model wrapped in the source-model interface.

    Carries its directory as ``model_id``, because a manifest records the source id and
    refuses to describe an artifact whose source it cannot name.
    """
    from transformers import AutoModel

    from byteify.source import HuggingFaceSourceEncoder

    return HuggingFaceSourceEncoder(
        AutoModel.from_pretrained(tiny_source), model_id=str(tiny_source)
    ).eval()


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizers):
    """The synthetic byte-level BPE tokenizer, matching the anchor model's tokenizer."""
    return tiny_tokenizers["bytelevel"]


@pytest.fixture(scope="session")
def tiny_source_tokenizer(tiny_tokenizers):
    """The tokenizer the synthetic source model was built alongside."""
    return tiny_tokenizers[SYNTHETIC_MODEL_TOKENIZER["modernbert"]]