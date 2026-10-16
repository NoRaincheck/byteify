"""Caches keyed on object identity.

Byteification resolves a tokenizer's whole vocabulary and builds a prefix trie, both of
which walk the entire vocabulary. Doing that per training example dominated preprocessing,
so both results are cached per tokenizer.

The cache key is the subtlety. Python recycles ``id()`` after garbage collection, so a
dictionary keyed on ``id()`` can hand one tokenizer's vocabulary to a *different* tokenizer
allocated at the same address. Every script in this repository builds exactly one tokenizer
per process, so it never bit. A tool that byteifies several models in one process has to
survive it -- and the address genuinely is recycled, most of the time, within a few
allocations.
"""

from __future__ import annotations

import gc

import pytest

from byteify.stitching import tokenizer_tables
from byteify.vocab import build_suffix_trie, resolve_token_bytes
from tests.synthetic import build_tokenizer

_ALPHA = [f"alpha {i} shared vocabulary words" for i in range(60)]
_OMEGA = [f"omega {i} entirely different words" for i in range(60)]


def test_a_recycled_address_does_not_serve_the_previous_tokenizers_vocabulary():
    """The bug this cache has to not have.

    Prime the cache with one tokenizer, drop it, and allocate until a new tokenizer lands
    on the recycled address. An identity-keyed cache still holds the dead tokenizer's
    entry under that address and hands it to its replacement.
    """
    first = build_tokenizer("bytelevel", _ALPHA)
    expected_first = resolve_token_bytes(first, use_cache=False)
    resolve_token_bytes(first)  # prime
    address = id(first)
    del first
    gc.collect()

    replacement = None
    for _ in range(60):
        replacement = build_tokenizer("bytelevel", _OMEGA)
        if id(replacement) == address:
            break
        del replacement
        replacement = None
        gc.collect()
    if replacement is None:
        pytest.skip("could not force an address collision on this allocator")

    got = resolve_token_bytes(replacement)
    assert got.token_bytes == resolve_token_bytes(replacement, use_cache=False).token_bytes
    assert got.token_bytes != expected_first.token_bytes, (
        "the replacement tokenizer was served the dead one's vocabulary"
    )


def test_every_tokenizer_resolves_to_its_own_vocabulary():
    """The general property: caching must never change an answer."""
    for texts in ([f"alpha {i} a" for i in range(60)], [f"beta {i} b" for i in range(60)]):
        tokenizer = build_tokenizer("bytelevel", texts)
        cached = resolve_token_bytes(tokenizer)
        assert cached.token_bytes == resolve_token_bytes(tokenizer, use_cache=False).token_bytes
        assert cached.token_bytes == resolve_token_bytes(tokenizer).token_bytes


def test_the_cached_trie_matches_a_freshly_built_one():
    tokenizer = build_tokenizer("bytelevel", [f"word {i} bytes" for i in range(80)])
    _, cached_trie, _ = tokenizer_tables(tokenizer)
    fresh_trie, _ = build_suffix_trie(tokenizer)

    data = [119, 111, 114, 100]
    end = len(data) - 1  # inclusive, per the trie's contract
    assert cached_trie.longest_suffix_id(data, end) == fresh_trie.longest_suffix_id(data, end)