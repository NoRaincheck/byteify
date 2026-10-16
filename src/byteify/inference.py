"""Inference: turning text into embeddings, and exporting to ONNX.

This is the path the rest of the package exists to enable. Training-time code may pool
patches however it likes; inference has two extra constraints:

**The patch axis must have a static length.** Training pools patches wherever the
boundary predictor fires, so the number of patches varies per input. ONNX graphs need
fixed output shapes, and the downstream attention over patches cannot start until the
patch sequence exists. We therefore use **budgeted patching**: score every byte by
boundary probability, keep the ``n_patches`` highest-scoring positions as patch ends, and
mask out the ones that fell below threshold. This is `TopK` + `Sort`, both ONNX
operators, so the whole thing traces.

Budgeted patching is not only an export workaround. It is a coherent inference rule --
"spend at most this many patches, choosing the boundaries you are most confident about" --
and it degrades gracefully in both directions: if the predictor is confident about few
boundaries the surplus slots are masked from pooling, and if it is confident about more
than the budget, the extra ones are dropped and their patches merge.

**Embeddings must be pooled consistently.** The model's sentence embedding is mean
pooling after the source encoder's output norm, matching how the source model and
``sentence-transformers`` pool. Padding patches are excluded from the mean.

The ONNX export traces the *whole* embedding path: byte embedding, local encoder,
boundary predictor, budgeted pooling, the frozen global model, output norm, mean pool.
Inputs are the raw byte ids, the per-byte suffix-token ids, and the byte mask. No
tokenizer is required at inference -- the tokenizer was needed to build ``expanded_ids``
during *training*; for deployment these come from the byte stream alone, see
:func:`byte_ids_for_inference`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def budgeted_patches_mask(mask: Tensor) -> Tensor:
    """bool patch mask -> long attention mask, for the global model's forward."""
    return mask.long()


@dataclass
class BudgetedPatches:
    """A fixed-length patch structure derived from boundary probabilities."""

    end_indices: Tensor  # [B, P] byte index of each patch's last byte, ascending
    mask: Tensor  # [B, P] bool; True where the patch is a real patch
    scores: Tensor  # [B, P] boundary probability at each selected position


def select_budgeted_patches(
    boundary_probs: Tensor,
    byte_mask: Tensor,
    n_patches: int,
    threshold: float = 0.5,
) -> BudgetedPatches:
    """Keep the ``n_patches`` most confident boundaries, in positional order.

    boundary_probs: [B, N], byte_mask: [B, N]

    Padding positions are pushed to -inf so they can never be selected as a patch end.
    The final real byte of each row is forced to a high score, because a patch sequence
    must be terminated -- otherwise the last patch would be unterminated and the text
    after the last chosen boundary would be dropped.
    """
    N = int(boundary_probs.shape[1])
    if n_patches > N:
        raise ValueError(f"n_patches={n_patches} exceeds the byte axis length {N}")

    scores = boundary_probs.masked_fill(~byte_mask.bool(), float("-inf"))

    # Force a boundary at the last real byte of every row.
    lengths = byte_mask.long().sum(dim=1).clamp_min(1)
    last_index = (lengths - 1).unsqueeze(1)
    # Scatter rather than index-assign (`scores[rows, last_index] = 2.0`): advanced
    # indexing by a tensor is not differentiable on every backend, and this sits
    # inside a training path.
    sentinel = torch.full_like(last_index, 2.0, dtype=scores.dtype)
    scores = scores.scatter(1, last_index, sentinel)

    _, indices = torch.topk(scores, k=n_patches, dim=1)
    # topk returns descending-score order; patches must be in positional order, so
    # re-sort the selected indices ascending and re-read their scores at those indices.
    indices, _ = torch.sort(indices, dim=1)
    values = torch.gather(scores, 1, indices)

    mask = (values > threshold) & torch.gather(byte_mask.bool(), 1, indices)
    # The terminating patch is always real, whatever the probability model said.
    # last_index is already [B, 1], so do not unsqueeze again: that broadcasts
    # [B, P] against [B, 1, 1] and silently yields [B, B, P].
    mask = mask | (indices == last_index)
    return BudgetedPatches(end_indices=indices, mask=mask, scores=values)


def pool_budgeted_patches(byte_reps: Tensor, patches: BudgetedPatches) -> Tensor:
    """Last-byte-of-patch pooling with a fixed patch axis. byte_reps: [B, N, D]."""
    idx = patches.end_indices.unsqueeze(-1).expand(-1, -1, byte_reps.shape[-1])
    reps = byte_reps.gather(1, idx)
    return reps * patches.mask.unsqueeze(-1).to(reps.dtype)


def byte_ids_for_inference(tokenizer, text: str, max_bytes: int = 512) -> tuple[Tensor, Tensor, Tensor]:
    """Raw-byte inputs for inference: ``(byte_ids, expanded_ids, byte_mask)``.

    Each is shaped ``[1, n_bytes]`` -- a single row, ready for the caller to pad and
    stack into a batch.

    ``expanded_ids`` is the per-byte suffix-token id that the local encoder's
    ``T_SubwordSuffix`` term adds to each byte embedding. During training it comes from
    the alignment of the source tokenizer; at inference we do not need the subword
    tokenizer's *segmentation*, only its *vocabulary*, so a reverse trie walk over the byte
    stream recovers the same ids. That keeps deployment free of any tokenizer call.
    """
    from .alignment import align_text, discover_special_bytes
    from .expand import expand_byte_ids

    specials = discover_special_bytes(tokenizer)
    from .stitching import tokenizer_tables

    _, trie, _ = tokenizer_tables(tokenizer)
    pseudo_to_token = {p: t for t, p in specials.items()}

    raw = text.encode("utf-8")[:max_bytes]
    decoded = raw.decode("utf-8", errors="ignore")
    a = align_text(decoded, tokenizer, special_token_ids=specials, add_special_tokens=False)
    byte_ids = torch.tensor(list(a.byte_ids), dtype=torch.long).unsqueeze(0)
    expanded = torch.tensor(
        expand_byte_ids(list(a.byte_ids), trie, pseudo_byte_to_token=pseudo_to_token),
        dtype=torch.long,
    ).unsqueeze(0)
    byte_mask = torch.ones_like(byte_ids)
    return byte_ids, expanded, byte_mask


class ByteifiedEmbedder(nn.Module):
    """The deployable artifact: text in, sentence embedding out.

    ``n_patches`` fixes the patch axis and therefore the whole ONNX graph shape. Choose it
    at or slightly above the expected bytes-per-patch ratio for the deployment domain;
    too low silently merges patches, too high wastes compute on masked slots.
    """

    def __init__(
        self,
        encoder: nn.Module,
        source,
        *,
        n_patches: int = 128,
        threshold: float = 0.5,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.source = source
        self.n_patches = n_patches
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, byte_ids: Tensor, expanded_ids: Tensor, byte_mask: Tensor) -> Tensor:
        n = min(self.n_patches, byte_ids.shape[1])
        embedding = _embedder_core(
            self.encoder, self.source, byte_ids, expanded_ids, byte_mask, n, self.threshold
        )
        return F.normalize(embedding, dim=-1) if self.normalize else embedding

    @torch.no_grad()
    def encode(self, tokenizer, texts: list[str], max_bytes: int = 512) -> Tensor:
        """Convenience path that also builds the byte inputs from a tokenizer."""
        byte_ids, expanded, mask = [], [], []
        for text in texts:
            b, e, m = byte_ids_for_inference(tokenizer, text, max_bytes)
            byte_ids.append(b)
            expanded.append(e)
            mask.append(m)
        width = max(x.shape[1] for x in byte_ids)
        pad = lambda xs, v: torch.stack([F.pad(x, (0, width - x.shape[1]), value=v) for x in xs])
        return self.forward(
            pad(byte_ids, 0), pad(expanded, -1), torch.cat(mask, dim=1)
        )


def _embedder_core(
    encoder: nn.Module,
    source,
    byte_ids: Tensor,
    expanded_ids: Tensor,
    byte_mask: Tensor,
    n_patches: int,
    threshold: float,
) -> Tensor:
    """The shared inference path, used by both eager forward and the ONNX wrapper.

    Kept as one function so the exported graph and the eager path cannot drift: they
    execute the same ops in the same order, which is what makes the export verifiable.
    """
    e = encoder.embed(byte_ids, expanded_ids)
    h = encoder.local_encoder(e, byte_mask)
    probs = torch.sigmoid(encoder.boundary_predictor(h))

    patches = select_budgeted_patches(probs, byte_mask, n_patches, threshold)
    patch_reps = pool_budgeted_patches(h, patches)

    hidden = source.forward_layers(patch_reps, patches.mask.long(), n_layers=source.n_layers)
    if source.final_norm is not None:
        hidden = source.final_norm(hidden)
    m = patches.mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)


def export_onnx(
    embedder: ByteifiedEmbedder,
    tokenizer,
    *,
    path: str = "byteified_embedder.onnx",
    max_bytes: int = 512,
    n_patches: int = 128,
    opset: int = 17,
    example_batch: tuple[Tensor, Tensor, Tensor] | None = None,
) -> str:
    """Trace the embedder to ONNX.

    Inputs are ``byte_ids``, ``expanded_ids`` and ``byte_mask``; the tokenizer is needed
    only to size and warm the graph, never at inference time. Pass ``example_batch`` to
    export for a specific batch shape (e.g. one captured during verification).
    """
    embedder.eval()
    if example_batch is None:
        example = "The quick brown fox jumps over the lazy dog. " * 4
        byte_ids, expanded, mask = byte_ids_for_inference(tokenizer, example, max_bytes)
    else:
        byte_ids, expanded, mask = example_batch

    # Three inputs must agree on the byte axis. Passing a differently-padded mask than
    # ids makes the local encoder see [B, 1, n_bytes, d_model] instead of [B, n_bytes,
    # d_model], because the two tensors then disagree on length and the downstream
    # reshape broadcasts. Pad defensively so callers cannot trip this.
    n_bytes = int(byte_ids.shape[1])
    if int(expanded.shape[1]) != n_bytes or int(mask.shape[1]) != n_bytes:
        raise ValueError(
            f"byte_ids, expanded_ids and byte_mask must share a byte axis; got "
            f"{n_bytes}, {int(expanded.shape[1])}, {int(mask.shape[1])}"
        )

    wrapper = _ExportWrapper(embedder, min(n_patches, n_bytes)).eval()
    # Only the batch axis is dynamic. The byte axis is deliberately fixed: the local
    # encoder's band mask needs a concrete sequence length, so callers pad their input
    # to `n_bytes` -- which is what a serving path does anyway to batch requests.
    # Output is named `sentence_embedding`, not `embedding`: the graph contains an
    # intermediate initializer for the nn.Embedding weight, and reusing the output name
    # makes ONNX Runtime reject the file with "Duplicate definition of name".
    dynamic_axes = {
        "byte_ids": {0: "batch"},
        "expanded_ids": {0: "batch"},
        "byte_mask": {0: "batch"},
    }
    # dynamo=True: the TorchScript tracer cannot handle this graph. It bakes traced
    # shapes into intermediate tensors, so the local encoder's [B, N, D] activations
    # arrive at attention as 4-D ([2, 1, 256, 256]) and the traced reshape then fails
    # with errors that name the wrong axis entirely. The dynamo exporter propagates
    # real shapes instead. It needs opset >= 18 for full dynamo support, so we raise
    # the floor rather than silently falling back.
    effective_opset = max(opset, 18)
    torch.onnx.export(
        wrapper,
        (byte_ids, expanded, mask),
        path,
        input_names=["byte_ids", "expanded_ids", "byte_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            **dynamic_axes,
            "sentence_embedding": {0: "batch"},
        },
        opset_version=effective_opset,
        dynamo=True,
    )
    return path


class _ExportWrapper(nn.Module):
    """Exposes the embedder's core as a 3-input ONNX-traceable module.

    ONNX graphs need static shapes and cannot branch on tensor values, so the patch
    budget is baked in as a Python constant here. That is exactly why
    :func:`select_budgeted_patches` exists: it turns a data-dependent patch count into a
    fixed one without losing the ability to mask out low-confidence slots.
    """

    def __init__(self, embedder: ByteifiedEmbedder, n_patches: int) -> None:
        super().__init__()
        self.embedder = embedder
        self.n_patches = n_patches

    def forward(self, byte_ids: Tensor, expanded_ids: Tensor, byte_mask: Tensor) -> Tensor:
        e = self.embedder
        out = _embedder_core(
            e.encoder, e.source, byte_ids, expanded_ids, byte_mask, self.n_patches, e.threshold
        )
        return F.normalize(out, dim=-1) if e.normalize else out