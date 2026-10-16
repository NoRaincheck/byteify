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
from .contrastive import (
    ContrastiveHead,
    ProjectionHead,
    infonce,
    mine_pairs,
    pair_batches,
    retrieval_metrics,
)
from .data import BatchSampler, iter_texts
from .expand import ByteTrie, expand_byte_ids
from .losses import (
    Stage1Weights,
    boundary_loss,
    embedding_loss,
    masked_byte_loss,
    pool,
    sample_mlm_mask,
    stitching_loss,
)
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
from .stitching import (
    ByteBatch,
    cos_sim,
    encode_texts,
    masked_l2,
    teacher_states,
    tokenizer_tables,
)
from .task_arithmetic import apply_delta, delta, task_arithmetic_merge
from .vocab import VocabByteResolution, build_suffix_trie, resolve_token_bytes

__all__ = [
    "SPECIAL_TOKENS",
    "tokenizer_tables",
    "teacher_states",
    "task_arithmetic_merge",
    "stitching_loss",
    "sample_mlm_mask",
    "retrieval_metrics",
    "pool",
    "pair_batches",
    "mine_pairs",
    "masked_l2",
    "masked_byte_loss",
    "iter_texts",
    "infonce",
    "encode_texts",
    "embedding_loss",
    "delta",
    "cos_sim",
    "boundary_loss",
    "apply_delta",
    "Stage1Weights",
    "ProjectionHead",
    "ContrastiveHead",
    "ByteBatch",
    "BatchSampler",
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