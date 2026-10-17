"""The byteified encoder: byte -> patches -> source encoder.

Architecture follows Bolmo's byteified LM (Fig. 1 of the Nature paper) with two
changes that the encoder setting makes natural:

* The **local encoder is bidirectional**. Bolmo's mLSTM must be causal because it
  has to run incrementally during decoding. An embedder has no decode step, so the
  local encoder can read the whole byte sequence in both directions.
* The **boundary predictor can use full context** for the same reason. Bolmo's
  cosine predictor is limited to 1 byte of lookahead because anything more would
  leak the next byte during generation. Here the 1-byte compromise is unnecessary,
  so the encoder-appropriate form is a plain MLP over the contextual byte
  representation. This fully closes the expressivity mismatch between a learned
  boundary predictor and a subword tokenizer, which is the specific defect the
  paper identifies.

What an embedder does *not* need, and what this module therefore omits: the local
decoder, the language-modelling head, the fused 512-entry byte/boundary vocabulary,
and the output boundary prediction that exists only to make generation exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: The score given to a masked-out key. Not ``-inf``: an all-``-inf`` row softmaxes to NaN,
#: and a padded query position is exactly such a row. ``finfo.min`` is finite, so the
#: softmax returns a finite (uniform) distribution that the caller then discards.
_MASKED_SCORE = {
    torch.float32: torch.finfo(torch.float32).min,
    torch.float16: torch.finfo(torch.float16).min,
    torch.bfloat16: torch.finfo(torch.bfloat16).min,
}

# Upper bound on the dense band mask built by ``LocalAttention`` (elements, not bytes).
# 2**28 elements is ~2 GB as bool and stays inside MPS's 32-bit indexing limit at the
# batch sizes this project uses.
MAX_DENSE_ATTENTION_ELEMENTS = 2**28


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Squeeze any trailing singleton. ONNX tracing propagates an extra axis through
        # the residual adds in LocalContextEncoder, so downstream modules would
        # otherwise receive [batch, n_bytes, d_model, 1] and misread the sequence
        # length. Eager execution never produces this, hence the guarded squeeze.
        if x.dim() == 4 and int(x.shape[3]) == 1:
            x = x.squeeze(3)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class ByteEmbedding(nn.Module):
    """``e_i = T_Byte(x_i) + T_SubwordSuffix(x_:i)``.

    The suffix term is what lets a very small local encoder stand in for the
    source model's entire input embedding matrix: the pretrained subword embeddings
    are simply broadcast onto every byte they cover. ``olmo_core`` calls the second
    table ``expanded_embeddings`` and initialises it from the source model's
    embedding matrix.
    """

    def __init__(self, n_byte_values: int, subword_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.byte = nn.Embedding(n_byte_values, d_model)
        self.suffix = nn.Embedding(subword_vocab_size, d_model)
        nn.init.normal_(self.byte.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.suffix.weight)

    def forward(self, byte_ids: Tensor, expanded_ids: Tensor) -> Tensor:
        """byte_ids, expanded_ids: [B, N] -> [B, N, d_model].

        ``expanded_ids`` holds the id of the longest vocabulary token ending at each
        byte; positions with no match (-1) fall back to a zero vector.
        """
        suffix = self.suffix(expanded_ids.clamp_min(0))
        suffix = suffix * (expanded_ids >= 0).unsqueeze(-1)
        return self.byte(byte_ids) + suffix

    @torch.no_grad()
    def load_suffix_from(self, embedding_weight: Tensor) -> None:
        """Initialise the suffix table from the source model's embedding matrix."""
        n = min(self.suffix.weight.shape[0], embedding_weight.shape[0])
        self.suffix.weight[:n].copy_(embedding_weight[:n].to(self.suffix.weight.dtype))


class LocalFeedForward(nn.Module):
    """SwiGLU feed-forward, matching ModernBERT's gated MLP."""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.Wi = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.silu(gate) * up)


class LocalAttention(nn.Module):
    """Bidirectional sliding-window self-attention over bytes.

    A portable stand-in for Bolmo's mLSTM local encoder: mLSTM was chosen for
    throughput during autoregressive decoding, but we need a module that runs on
    CPU/MPS and that can look in both directions. Sliding-window attention keeps the
    cost linear in sequence length at a fixed window.
    """

    def __init__(self, d_model: int, n_heads: int, window: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Materialise concrete ints once. Export fixes the byte length (see below), so
        # int() is safe, and every downstream shape expression then uses plain Python
        # ints instead of re-reading x.shape inside traced subexpressions.
        #
        # Several tracing traps are avoided by being explicit here:
        #   * `B, N, D = x.shape` raises "too many values to unpack" -- a traced shape
        #     carries an extra entry.
        #   * Re-reading x.shape[i] inside a generator expression yields
        #     "[2, 1, 4, 64] is invalid for input of size 131072".
        #   * The RMSNorm that precedes this module returns a 4-D tensor when traced
        #     through ONNX (it appends a trailing singleton), which previously made
        #     N read as 1 here.
        if x.dim() != 3:
            raise ValueError(f"LocalAttention expects [batch, n_bytes, d_model], got {tuple(x.shape)}")
        B, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])

        # The band mask is materialised densely as [B, 1, N, N], so memory grows as
        # B*N^2. Backends differ in the largest index they accept -- MPS rejects
        # anything needing 64-bit indexing, which this hits at a few thousand bytes --
        # so raise with the real numbers rather than an opaque backend error.
        n_elem = B * N * N
        if n_elem > MAX_DENSE_ATTENTION_ELEMENTS:
            raise ValueError(
                f"LocalAttention would build a {B}x1x{N}x{N} mask "
                f"({n_elem:,} elements). Reduce --batch-size or --max-bytes, or raise "
                f"MAX_DENSE_ATTENTION_ELEMENTS above {MAX_DENSE_ATTENTION_ELEMENTS:,}."
            )
        # NOTE: this module uses reshape rather than view throughout. Under ONNX tracing
        # the leading dimension is symbolic; view() then fails with "shape
        # '[2, 1, 4, 64]' is invalid" because it reads the traced constant 2 rather
        # than the symbolic batch. reshape() reinterprets the buffer and traces.
        qkv = self.Wqkv(x).chunk(3, dim=-1)
        # [B, N, d_model] -> [B, heads, N, head_dim]. Reshape + permute with the
        # concrete ints above; using the concrete B/N here is what keeps the traced
        # shape consistent.
        q, k, v = (
            t.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3) for t in qkv
        )

        # Banded attention, implemented with an explicit additive bias rather than the
        # fused kernel. PyTorch's SDPA squeezes singleton dims off the mask under ONNX
        # tracing, then fails to broadcast the [98, 4, 1, 1] result back to
        # [98, 4, 1, 93] and errors. The explicit matmul below has no such ambiguity,
        # and at these sizes the speed difference is not the bottleneck.
        #
        # The band mask requires a *concrete* sequence length. Under ONNX tracing with a
        # symbolic byte axis, torch.arange(...) produces a tensor of non-static rank and
        # every downstream shape breaks in ways that report the wrong axis ("size of
        # tensor a (4) must match b (93) at dimension 3"). Export therefore fixes the
        # byte length and callers pad to it; see byteify.inference.export_onnx. Eager
        # use is unaffected because shapes are always concrete there.
        n = int(x.shape[1])
        positions = torch.arange(n, device=x.device)
        offset = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N]
        allowed = (offset.abs() <= self.window) & (offset <= 0)
        mask = allowed.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(
            ~mask, _MASKED_SCORE[q.dtype]
        )
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        # A query position that is itself padding has no permitted keys at all -- every key
        # at or before it is padding too. softmax over an all-masked row is undefined and
        # returns NaN, which then propagates through the residual stream into every
        # parameter's gradient. Zero those rows instead: their output is discarded anyway,
        # and a NaN that only nobody reads is still a NaN that trains.
        query_valid = mask.any(dim=-1, keepdim=True)
        out = out * query_valid.to(out.dtype)
        # [B, heads, N, head_dim] -> [B, N, d_model]. permute then reshape, mirroring
        # the split above. (flatten(1, 2) is not a substitute: it merges heads into the
        # sequence axis and silently yields [B, N * heads, head_dim].)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.Wo(out)


class LocalGRU(nn.Module):
    """Bidirectional GRU over bytes; an even cheaper portable local encoder."""

    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            d_model,
            d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


class LocalContextEncoder(nn.Module):
    """The local encoder E: a few bidirectional layers over bytes.

    Bolmo uses a single mLSTM layer for the encoder because the retained subword
    embeddings supply the model with most of the input representation. We keep that
    asymmetry (fewer layers than a decoder would need) and put the capacity into the
    feed-forward width instead.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        ffn_hidden: int = 1024,
        n_heads: int = 4,
        window: int = 128,
        backend: Literal["attention", "gru"] = "attention",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.attn_norms = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(n_layers):
            if backend == "attention":
                self.attns.append(LocalAttention(d_model, n_heads, window))
                self.attn_norms.append(RMSNorm(d_model))
            self.ffns.append(LocalFeedForward(d_model, ffn_hidden))
            self.ffn_norms.append(RMSNorm(d_model))
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        for i, ffn in enumerate(self.ffns):
            if self.backend == "attention":
                x = x + self.attns[i](self.attn_norms[i](x), attention_mask)
            x = x + ffn(self.ffn_norms[i](x))
        return self.out_norm(x)


class CosineBoundaryPredictor(nn.Module):
    """Bolmo's boundary predictor: cosine distance between projected byte ``t`` and byte ``t+1``.

    Kept for comparison with the paper. Its 1-byte lookahead exists to avoid leaking
    the next byte during decoding; for an embedder `MLPBoundaryPredictor` is the
    better default.
    """

    def __init__(self, d_model: int, lookahead: int = 1) -> None:
        super().__init__()
        self.lookahead = lookahead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D] -> boundary logits [B, N]."""
        N = x.shape[1]
        la = self.lookahead
        q = F.normalize(self.q_proj(x[:, : N - la]), dim=-1)
        k = F.normalize(self.k_proj(x[:, la:]), dim=-1)
        cos = (q * k).sum(-1)
        score = 0.5 * (1.0 - cos)  # in [0, 1]; high == boundary
        # The last `lookahead` bytes have no successor to compare against. A real
        # boundary can still land there, so score them as "no boundary" and let the
        # BCE loss handle the tail.
        return F.pad(score, (0, la), value=0.0)


class MLPBoundaryPredictor(nn.Module):
    """Full-context boundary predictor: the encoder-appropriate form.

    ``p_t`` depends on the whole byte sequence via the local encoder, so it can
    realise exactly the non-causal boundary decisions a subword tokenizer makes
    (e.g. committing to "flower" only once the next byte shows "bed"). Returns
    logits for binary cross-entropy against ``B_subword``.
    """

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def patch_index(boundaries: Tensor) -> Tensor:
    """Index of the patch each byte belongs to. boundaries: [B, N] bool -> [B, N]."""
    return (boundaries.long().cumsum(dim=1) - 1).clamp_min(0)


def pool_patches(byte_reps: Tensor, boundaries: Tensor) -> tuple[Tensor, Tensor]:
    """Bolmo/H-Net pooling: take the representation of the *last* byte of each patch.

    byte_reps: [B, N, D], boundaries: [B, N] bool (1 at patch ends)
    returns patch_reps [B, P, D] and patch_mask [B, P] bool.

    Padding is handled by only writing at positions that actually close a patch.
    A ``scatter_`` over all byte positions would be wrong here: padded positions
    carry the same patch index as the sequence's final patch and would clobber the
    real pooled vector, with which of the two wins being undefined.
    """
    B, _, D = byte_reps.shape
    idx = patch_index(boundaries)
    P = int(boundaries.sum(dim=1).max().item())

    patch_reps = byte_reps.new_zeros(B, P, D)
    patch_mask = torch.zeros(B, P, dtype=torch.bool, device=byte_reps.device)

    rows, cols = boundaries.nonzero(as_tuple=True)
    targets = idx[rows, cols]
    patch_reps[rows, targets] = byte_reps[rows, cols]
    patch_mask[rows, targets] = True
    return patch_reps, patch_mask


def depool(depooled_resid: Tensor, patch_reps: Tensor, boundaries: Tensor) -> Tensor:
    """Bolmo/H-Net depooling: add the latest available patch representation to a linear
    projection of the byte representations, giving one representation per byte again.

    patch_reps: [B, P, D], boundaries: [B, N] bool (1 at patch ends)
    returns [B, N, D]
    """
    B, P, D = patch_reps.shape
    N = boundaries.shape[1]
    idx = patch_index(boundaries)  # [B, N] patch each byte belongs to
    out = patch_reps.new_zeros(B, N, D)
    valid = idx < P
    safe_idx = idx.clamp(max=max(P - 1, 0))
    if P > 0:
        gathered = patch_reps.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, D))
        out = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return depooled_resid + out


class Depooling(nn.Module):
    """Project byte representations, add the latest patch representation, return bytes."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, byte_reps: Tensor, patch_reps: Tensor, boundaries: Tensor) -> Tensor:
        return depool(self.proj(byte_reps), patch_reps, boundaries)


class ByteMLMHead(nn.Module):
    """Predict a masked byte from its contextualised representation.

    The encoder analogue of Bolmo's language-modelling head. Two differences matter:

    * Bolmo predicts the **next** byte; a bidirectional encoder has no next byte, so the
      Stage 2 objective is **masked** byte prediction.
    * Bolmo's vocabulary is 512 entries (each byte fused with a boundary flag) so that
      boundary prediction is free at sampling time. There is no sampling here, so this
      head covers the 256 real bytes only. Pseudo-bytes standing in for ``[CLS]``-style
      tokens are never a prediction target.
    """

    def __init__(self, d_model: int, n_bytes: int = 256) -> None:
        super().__init__()
        self.n_bytes = n_bytes
        self.proj = nn.Linear(d_model, n_bytes)

    def forward(self, byte_context: Tensor) -> Tensor:
        """byte_context: [B, N, D] -> logits over real bytes, [B, N, 256]."""
        return self.proj(byte_context)


class PartialModernBert(nn.Module):
    """Run the first ``n`` layers of a ModernBERT encoder over given representations.

    Stage 1's ``L_E`` loss compares the student's representations at layer ``n``
    against the teacher's, so we need to truncate the frozen global model.
    ``inputs_embeds`` already skips the token-embedding lookup but keeps the
    embedding norm, which is exactly the point the stitching loss compares at.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.n_layers = len(model.layers)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, n_layers: int | None = None) -> Tensor:
        from transformers.masking_utils import (
            create_bidirectional_mask,
            create_bidirectional_sliding_window_mask,
        )

        n = self.n_layers if n_layers is None else n_layers
        model = self.model
        h = model.embeddings(inputs_embeds=inputs_embeds)

        B, N, _ = h.shape
        device = h.device
        position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": h,
            "attention_mask": attention_mask,
        }
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        pos = {
            layer_type: model.rotary_emb(h, position_ids, layer_type)
            for layer_type in set(model.config.layer_types)
        }
        for layer in model.layers[:n]:
            h = layer(
                h,
                attention_mask=masks[layer.attention_type],
                position_embeddings=pos[layer.attention_type],
            )
        return h


@dataclass
class ByteifiedEncoderOutput:
    byte_reps: Tensor
    patch_reps: Tensor
    patch_mask: Tensor
    boundary_logits: Tensor | None = None
    byte_context: Tensor | None = None  # post-depool + local decoder, byte axis


class ByteifiedEncoder(nn.Module):
    """Bytes in, source-encoder patch representations out."""

    def __init__(
        self,
        d_model: int,
        n_byte_values: int,
        subword_vocab_size: int,
        local_encoder: LocalContextEncoder | None = None,
        boundary_predictor: Literal["mlp", "cosine", "none"] = "mlp",
        local_decoder_layers: int = 0,
        local_decoder_ffn_hidden: int | None = None,
        local_decoder_window: int = 128,
        n_heads: int = 4,
        backend: Literal["attention", "gru"] = "attention",
    ) -> None:
        super().__init__()
        self.embed = ByteEmbedding(n_byte_values, subword_vocab_size, d_model)
        self.local_encoder = local_encoder or LocalContextEncoder(d_model)
        self.boundary_kind = boundary_predictor
        if boundary_predictor == "mlp":
            self.boundary_predictor: nn.Module | None = MLPBoundaryPredictor(d_model, d_model)
        elif boundary_predictor == "cosine":
            self.boundary_predictor = CosineBoundaryPredictor(d_model)
        else:
            self.boundary_predictor = None

        # Optional output side. Bolmo's local decoder exists to produce next-byte
        # logits; here it turns depooled patch representations back into contextualised
        # byte representations, which Stage 2 needs for byte-level MLM and for pooling
        # the embedding over bytes rather than patches.
        self.depool: nn.Module | None = None
        self.local_decoder: nn.Module | None = None
        if local_decoder_layers > 0:
            self.depool = Depooling(d_model)
            self.local_decoder = LocalContextEncoder(
                d_model,
                n_layers=local_decoder_layers,
                ffn_hidden=local_decoder_ffn_hidden or 1024,
                n_heads=n_heads,
                window=local_decoder_window,
                backend=backend,
            )

    def forward(
        self,
        byte_ids: Tensor,
        expanded_ids: Tensor,
        byte_mask: Tensor,
        boundaries: Tensor | None = None,
    ) -> ByteifiedEncoderOutput:
        """byte_ids/expanded_ids/byte_mask/boundaries: [B, N].

        When ``boundaries`` is given the *oracle* subword boundaries are used for
        pooling. That is what Stage 1's ``L_E`` requires: the paper pools with
        ``B_subword`` so the student's patch axis stays aligned with the teacher's
        token axis. At inference the learned boundary predictor supplies them.
        """
        e = self.embed(byte_ids, expanded_ids)
        h = self.local_encoder(e, byte_mask)

        boundary_logits = None
        if self.boundary_predictor is not None:
            boundary_logits = self.boundary_predictor(h)

        used = boundaries
        if used is None:
            if boundary_logits is None:
                raise ValueError("need either oracle boundaries or a boundary predictor")
            used = torch.sigmoid(boundary_logits) > 0.5
            # Guarantee a terminated final patch (on the last real byte of each row).
            last = byte_mask.flip(1).argmax(dim=1)
            used[torch.arange(used.shape[0], device=used.device), last] = True

        patch_reps, patch_mask = pool_patches(h, used.bool())

        byte_context = None
        if self.depool is not None and self.local_decoder is not None:
            z = self.depool(h, patch_reps, used.bool())
            byte_context = self.local_decoder(z, byte_mask)

        return ByteifiedEncoderOutput(h, patch_reps, patch_mask, boundary_logits, byte_context)