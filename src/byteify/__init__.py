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
from .inference import (
    BudgetedPatches,
    ByteifiedEmbedder,
    budgeted_patches_mask,
    export_onnx,
    pool_budgeted_patches,
    select_budgeted_patches,
)
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
from .source import (
    ByteifyConfig,
    LocalEncoderSpec,
    SourceEncoder,
    build_byteified_encoder,
    load_byteified_encoder,
    resolve_source_encoder,
)
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
    "Alignment",
    "BatchSampler",
    "BudgetedPatches",
    "ByteBatch",
    "ByteEmbedding",
    "ByteTrie",
    "ByteifiedEmbedder",
    "ByteifiedEncoder",
    "ByteifiedEncoderOutput",
    "ByteifyConfig",
    "ContrastiveHead",
    "CosineBoundaryPredictor",
    "Depooling",
    "LocalAttention",
    "LocalContextEncoder",
    "LocalEncoderSpec",
    "LocalFeedForward",
    "LocalGRU",
    "MLPBoundaryPredictor",
    "PartialModernBert",
    "ProjectionHead",
    "RMSNorm",
    "SourceEncoder",
    "Stage1Weights",
    "VocabByteResolution",
    "align",
    "align_from_offsets",
    "align_text",
    "apply_delta",
    "boundary_loss",
    "bpe_merge_patches",
    "budgeted_patches_mask",
    "build_byteified_encoder",
    "build_suffix_trie",
    "bytes_to_unicode",
    "cos_sim",
    "delta",
    "discover_special_bytes",
    "embedding_loss",
    "encode_texts",
    "expand_byte_ids",
    "export_onnx",
    "infonce",
    "iter_texts",
    "load_byteified_encoder",
    "masked_byte_loss",
    "masked_l2",
    "merge_batch",
    "merge_boundaries",
    "mine_pairs",
    "pair_batches",
    "patch_index",
    "pool",
    "pool_budgeted_patches",
    "pool_patches",
    "resolve_source_encoder",
    "resolve_token_bytes",
    "retrieval_metrics",
    "sample_mlm_mask",
    "select_budgeted_patches",
    "stitching_loss",
    "task_arithmetic_merge",
    "teacher_states",
    "tokenizer_tables",
]