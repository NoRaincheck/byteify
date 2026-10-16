"""byteify: retrofitting subword embedding models to operate over raw bytes.

Port of Bolmo's byteification (Minixhofer et al., *Nature* 2026,
doi:10.1038/s41586-026-11111-4) from autoregressive language models to
*embedding* models, anchored on `jhu-clsp/ettin-encoder-17m`.

Bolmo's method has no encoder counterpart upstream: `allenai/bolmo-core` is
decoder-only throughout (it mandates an LM head, its local decoder exists only for
next-byte prediction, and `olmo-core` has no ModernBERT). This package reuses the
*algorithm* and rebuilds the parts that are encoder-specific.
"""

from .alignment import (
    SPECIAL_TOKENS,
    Alignment,
    align,
    align_from_offsets,
    align_text,
    bytes_to_unicode,
    discover_special_bytes,
)
from .boundary_merge import bpe_merge_patches, merge_batch, merge_boundaries
from .expand import ByteTrie, expand_byte_ids
from .model import (
    ByteEmbedding,
    ByteifiedEncoder,
    ByteifiedEncoderOutput,
    CosineBoundaryPredictor,
    Depooling,
    LocalAttention,
    LocalContextEncoder,
    LocalFeedForward,
    LocalGRU,
    MLPBoundaryPredictor,
    PartialModernBert,
    RMSNorm,
    patch_index,
    pool_patches,
)
from .source import ByteifyConfig, SourceEncoder, resolve_source_encoder
from .vocab import VocabByteResolution, build_suffix_trie, resolve_token_bytes

__all__ = [
    "SPECIAL_TOKENS",
    "Alignment",
    "ByteEmbedding",
    "ByteTrie",
    "ByteifiedEncoder",
    "ByteifiedEncoderOutput",
    "ByteifyConfig",
    "CosineBoundaryPredictor",
    "Depooling",
    "LocalAttention",
    "LocalContextEncoder",
    "LocalFeedForward",
    "LocalGRU",
    "MLPBoundaryPredictor",
    "PartialModernBert",
    "RMSNorm",
    "SourceEncoder",
    "VocabByteResolution",
    "align",
    "align_from_offsets",
    "align_text",
    "build_suffix_trie",
    "bpe_merge_patches",
    "bytes_to_unicode",
    "discover_special_bytes",
    "expand_byte_ids",
    "merge_batch",
    "merge_boundaries",
    "resolve_token_bytes",
]