"""Loading a byteified artifact through its manifest, and nothing else.

This is the seam the product is judged on: take a directory, hand it text, get an
embedding, with no other configuration supplied and no source tokenizer anywhere in the
path.

The source *model* is still fetched, because a byteified encoder delegates its patch
representations to the frozen global encoder it was distilled from -- that delegation is
what level 1's fidelity guarantee *is*. The source *tokenizer* is what byteify removes,
and the artifact does not need one: the vocabulary byte strings its suffix trie is built
from ship inside the directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .expand import ByteTrie
from .source import (
    LocalEncoderSpec,
    build_encoder_from_dimensions,
    load_encoder_state,
    resolve_source_encoder,
)

if TYPE_CHECKING:
    from .inference import ByteifiedEmbedder
    from .manifest import Manifest


def build_embedder_from_manifest(
    manifest: Manifest,
    artifact_dir: Path | str,
    *,
    device: str = "cpu",
) -> ByteifiedEmbedder:
    """Construct a :class:`ByteifiedEmbedder` from a manifest and the artifact directory.

    The encoder's architecture comes from ``manifest.encoder_arch`` and its dimensions
    from ``manifest.source_d_model`` / ``n_byte_values`` / ``vocab_size`` -- never from
    constants at this call site. That was the defect this replaces: an export path that
    hardcoded ``n_layers=1, ffn_hidden=1024, window=128`` could not export a checkpoint
    trained with anything else, and the mismatch surfaced as a shape error on load.
    """
    import torch

    from .inference import ByteifiedEmbedder
    from .manifest import ENCODER_NAME

    artifact_dir = Path(artifact_dir)
    source = resolve_source_encoder(manifest.source_id).to(device)
    source.eval()

    encoder = build_encoder_from_dimensions(
        d_model=manifest.source_d_model,
        n_byte_values=manifest.n_byte_values,
        subword_vocab_size=manifest.vocab_size,
        spec=LocalEncoderSpec.from_config(manifest.encoder_arch),
        n_heads=getattr(source, "n_attention_heads", 8),
    ).to(device)

    weights_path = artifact_dir / ENCODER_NAME
    if not weights_path.exists():
        raise FileNotFoundError(
            f"no {ENCODER_NAME} in {artifact_dir}; the manifest describes an encoder whose "
            "weights are not here"
        )
    load_encoder_state(encoder, torch.load(weights_path, map_location=device, weights_only=True))
    encoder.eval()

    # pooling and normalize come from the manifest rather than from byteify's defaults:
    # they are what the source repository declared, they are baked into the exported
    # graph, and serving with different ones would return a vector space the gates never
    # graded. A manifest predating them carries neither field, so it reads back as mean
    # pooling with normalisation -- what every artifact through schema 1.0 described.
    embedder = ByteifiedEmbedder(
        encoder,
        source,
        n_patches=manifest.budget,
        pooling=manifest.pooling,
        normalize=manifest.normalize,
    ).to(device)

    # The trie and the byte-stream rule are what turn text into graph inputs with no
    # tokenizer present, so the embedder carries them and can be handed text directly.
    embedder.byte_trie = ByteTrie(read_token_bytes(artifact_dir / manifest.trie.path))
    embedder.byte_stream_rule = manifest.stream_rule()
    embedder.byte_axis = manifest.max_bytes
    return embedder


def read_token_bytes(path: Path) -> dict[int, bytes]:
    from .manifest import read_trie

    return read_trie(path)