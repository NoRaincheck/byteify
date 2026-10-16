"""Boundary merging (the paper's compression-factor experiment)."""

from math import ceil

import pytest

from byteify.boundary_merge import bpe_merge_patches
from byteify.boundary_merge import merge_boundaries as merge_indicator


def test_ceil():
    assert ceil(4.0) == 4
    assert ceil(4.2) == 5
    assert ceil(0.1) == 1


def test_no_merge_when_target_is_met():
    lengths = [4, 4, 4, 4]
    keep = bpe_merge_patches([10, 11, 12, 13], lengths, target_bytes_per_patch=4.0)
    assert keep == [True] * 4


def test_merge_reaches_target_ratio():
    lengths = [4] * 16  # 64 bytes, 16 patches
    ids = list(range(1000, 1000 + 16))
    keep = bpe_merge_patches(ids, lengths, target_bytes_per_patch=8.0)
    assert sum(keep) == 8, "64 bytes at 8 bytes/patch is 8 patches"
    # Bytes are conserved exactly when boundaries are dropped.
    merged = merge_indicator([0, 1] * 16, keep)
    assert sum(merged) == 8


def test_merge_never_drops_the_final_boundary():
    """One patch is unreachable: the final boundary has to survive to terminate it."""
    lengths = [4] * 10  # 40 bytes; a target of 64 would ask for a single patch
    keep = bpe_merge_patches(list(range(10)), lengths, target_bytes_per_patch=64.0)
    merged = merge_indicator([0, 1] * 10, keep)
    assert list(merged)[-1] == 1
    assert sum(merged) == 2, "two patches is the minimum when the last boundary is kept"


def test_merge_reaches_the_requested_ratio():
    """32 bytes at a target of 8 bytes/patch is 4 patches."""
    ids = [1, 2] * 8  # A B A B ... so pair (1,2) is by far the most frequent
    lengths = [2] * 16
    keep = bpe_merge_patches(ids, lengths, target_bytes_per_patch=8.0)
    assert sum(keep) == 4


def test_merge_is_deterministic():
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
    lengths = [3] * len(ids)
    a = bpe_merge_patches(ids, lengths, 6.0)
    b = bpe_merge_patches(ids, lengths, 6.0)
    assert a == b


def test_merge_on_all_distinct_tokens_still_reaches_target():
    ids = list(range(1000, 1016))
    lengths = [2] * 16
    keep = bpe_merge_patches(ids, lengths, target_bytes_per_patch=6.0)
    assert sum(keep) == ceil(32 / 6.0)


def test_merge_indicator_validates_length():
    with pytest.raises(ValueError):
        merge_indicator([0, 1, 0, 1], keep=[True])


def test_merge_indicator_applies_mask():
    boundaries = [0, 1, 0, 1, 0, 1]
    assert merge_indicator(boundaries, [True, False, True]) == [0, 1, 0, 0, 0, 1]


def test_single_patch_never_merged():
    assert bpe_merge_patches([5], [10], 1.0) == [True]


def test_empty_input():
    assert bpe_merge_patches([], [], 4.0) == []