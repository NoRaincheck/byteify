"""Increasing the compression factor by merging patch boundaries.

A byteified model is not restricted to the fixed patch set of a subword tokenizer, so
the boundary supervision ``B_subword`` can be *thinned*: dropping some boundaries makes
patches longer, raises bytes-per-patch, and speeds the global model up at no softmax
cost. The paper reports this is the mechanism behind the efficiency result (Fig. 2),
and that -- unlike a subword model, which would need a larger vocabulary and pays a
growing softmax cost -- a byte model can be pushed to an arbitrarily high compression
ratio with a smooth drop-off in performance.

Three ways of choosing which boundaries to drop were compared. This module implements
the one the paper found best:

* **BPE** (here) -- iteratively merge the most frequent adjacent token pair, applied
  *per example* rather than corpus-wide, following Feher et al. The paper's explanation
  for why this wins specifically in the byteifying setting: BPE merging makes the
  fewest distinct merges to reach any target compression, so it stays closest to the
  behaviour the model was Stage-1 distilled from.
* **Entropy** and **cross-entropy** merging (paper) -- need an auxiliary subword LM to
  score candidate pairs. Not implemented; the paper finds both worse than BPE.

Only boundaries are dropped, so **no bytes are lost**: the merged boundary indicator
covers the same byte stream with fewer patches.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from math import ceil


def merge_boundaries(boundaries: Sequence[int], keep: Sequence[bool]) -> list[int]:
    """Apply a per-patch ``keep`` mask to a boundary indicator.

    Always keeps the final boundary: dropping it would leave the last patch
    unterminated and silently drop text from the patch sequence.
    """
    n_patches = sum(boundaries)
    if len(keep) != n_patches:
        raise ValueError("keep must have one entry per patch")
    out: list[int] = []
    idx = 0
    for b in boundaries:
        out.append(b)
        if b:
            if not keep[idx] and idx != n_patches - 1:
                out[-1] = 0
            idx += 1
    return out


def bpe_merge_patches(
    token_ids: Sequence[int],
    lengths: Sequence[int],
    target_bytes_per_patch: float,
) -> list[bool]:
    """Choose which patch boundaries survive, per example, via per-example BPE merging.

    Returns ``keep``, one entry per patch, where ``keep[i]`` says whether the boundary
    closing patch *i* is retained. Dropping boundary *i* fuses patch *i* into patch
    *i+1*, so bytes are conserved.

    Merges are chosen over token id pairs, as byte pair encoding does, which is what
    makes the number of distinct merges needed to reach a target minimal. Implemented as
    a lazy-deletion heap over a doubly linked list of patches, so cost is near-linear
    rather than quadratic in the number of patches.
    """
    n = len(lengths)
    if n == 0:
        return []
    keep = [True] * n
    target_patches = max(1, ceil(sum(lengths) / target_bytes_per_patch))
    if target_patches >= n:
        return keep

    # Doubly linked list of live patches; a merge unlinks the right node.
    prev = [i - 1 for i in range(n)]
    nxt = [i + 1 for i in range(n)]
    nxt[-1] = -1
    alive = [True] * n
    tok = [token_ids[i] if i < len(token_ids) else i for i in range(n)]

    counts: dict[tuple[int, int], int] = {}
    where: dict[tuple[int, int], set[int]] = {}
    heap: list[tuple[int, tuple[int, int]]] = []
    # Each merge mints a fresh symbol id, as BPE does. Storing the constituent pair
    # instead would nest tuples and make the pair keys uncomparable (and would let two
    # different merges collide onto the same symbol).
    next_symbol = max((abs(t) for t in tok), default=0) + 1

    def add_pair(left: int, right: int) -> None:
        if left == -1 or right == -1:
            return
        pair = (tok[left], tok[right])
        counts[pair] = counts.get(pair, 0) + 1
        where.setdefault(pair, set()).add(left)
        # Every increment needs a fresh heap entry: the pop loop discards entries whose
        # cached count no longer matches, relying on a newer entry existing.
        heapq.heappush(heap, (-counts[pair], pair))

    def drop_pair(left: int, right: int) -> None:
        if left == -1 or right == -1:
            return
        pair = (tok[left], tok[right])
        if counts.get(pair, 0) <= 0:
            return
        counts[pair] -= 1
        bucket = where.get(pair)
        if bucket is not None:
            bucket.discard(left)
            if not bucket:
                where.pop(pair, None)

    heap = []
    i = 0
    while i != -1:
        add_pair(i, nxt[i])
        i = nxt[i]
    heapq.heapify(heap)

    merged = 0
    target_merges = n - target_patches
    while merged < target_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        if -neg_count <= 0:
            break
        if counts.get(pair, 0) != -neg_count:
            continue  # stale entry; a newer one exists
        candidates = where.get(pair)
        if not candidates:
            continue
        left = next(iter(candidates))
        right = nxt[left]
        if right == -1 or (tok[left], tok[right]) != pair:
            candidates.discard(left)
            continue
        if nxt[right] == -1:
            # Dropping the final boundary is a no-op: merge_boundaries always keeps it.
            candidates.discard(left)
            continue

        # Detach the pair's contribution, then splice `right` out of the list.
        drop_pair(left, right)
        outer_r = nxt[right]
        drop_pair(right, outer_r)
        l_prev = prev[left]
        drop_pair(l_prev, left)

        alive[right] = False
        nxt[left] = outer_r
        if outer_r != -1:
            prev[outer_r] = left
        tok[left] = next_symbol
        next_symbol += 1
        keep[right] = False
        merged += 1

        add_pair(l_prev, left)
        add_pair(left, outer_r)

    return keep


def patch_lengths_from_boundaries(boundaries: Sequence[int]) -> list[int]:
    """Byte length of each patch, given a boundary indicator."""
    lengths: list[int] = []
    run = 0
    for b in boundaries:
        run += 1
        if b:
            lengths.append(run)
            run = 0
    return lengths


def merge_batch(batch, target_bytes_per_patch: float):
    """Merge boundaries across a whole batch, per example, as the paper does.

    Returns ``(merged_batch, achieved_bytes_per_patch)``.
    """
    import torch

    from .stitching import ByteBatch

    new_rows = []
    for b in range(batch.boundaries.shape[0]):
        n_bytes = int(batch.byte_mask[b].sum().item())
        boundaries = [int(x) for x in batch.boundaries[b, :n_bytes]]
        if not boundaries or sum(boundaries) == 0:
            new_rows.append(boundaries)
            continue
        lengths = patch_lengths_from_boundaries(boundaries)
        # Patch k corresponds to the k-th token that owns at least one byte.
        owning = [int(t) for t, l in zip(batch.input_ids[b], batch.lengths[b]) if l > 0]
        keep = bpe_merge_patches(owning[: len(lengths)], lengths, target_bytes_per_patch)
        new_rows.append(merge_boundaries(boundaries, keep))

    out = torch.zeros_like(batch.boundaries)
    for b, r in enumerate(new_rows):
        out[b, : len(r)] = torch.tensor(r, dtype=out.dtype)
    out = out * batch.byte_mask

    merged = ByteBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        byte_ids=batch.byte_ids,
        expanded_ids=batch.expanded_ids,
        byte_mask=batch.byte_mask,
        boundaries=out,
        lengths=batch.lengths,
    )
    achieved = int(batch.byte_mask.sum().item()) / max(int(out.sum().item()), 1)
    return merged, achieved