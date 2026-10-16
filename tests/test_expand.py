"""Tests for the suffix-embedding trie (``T_SubwordSuffix``)."""

import random
import time

import pytest

from byteify.expand import expand_byte_ids
from byteify.vocab import build_suffix_trie, resolve_token_bytes

MODEL_ID = "jhu-clsp/ettin-encoder-17m"


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def resolution(tokenizer):
    return resolve_token_bytes(tokenizer)


@pytest.fixture(scope="module")
def trie_and_specials(tokenizer):
    return build_suffix_trie(tokenizer)


@pytest.fixture(scope="module")
def trie(trie_and_specials):
    return trie_and_specials[0]


def brute_force_expand(byte_ids, resolution):
    """Reference: longest vocabulary entry that is a suffix of bytes[:i+1].

    Ties -- two entries with the same byte string, e.g. the BPE token "ĠĠ" and the
    added token "  " -- resolve to the higher id, matching the trie's last-insert-wins
    rule and olmo_core's trie construction.
    """
    vocab = {i: b for i, b in enumerate(resolution.token_bytes) if b}
    out = []
    for i in range(len(byte_ids)):
        best_id, best_len = -1, 0
        for tid, seq in vocab.items():
            n = len(seq)
            if n < best_len or n > i + 1:
                continue
            if n == best_len and tid <= best_id:
                continue
            if list(byte_ids[i - n + 1 : i + 1]) == list(seq):
                best_id, best_len = tid, n
        out.append(best_id)
    return out


def test_trie_max_depth(resolution, trie):
    longest = max(len(b) for b in resolution.token_bytes if b)
    assert trie.max_depth == longest


@pytest.mark.parametrize(
    "text",
    ["Hello World!", "unbelievable", "日本語", "the flowerbed", "  spaces  "],
)
def test_expand_matches_brute_force(text, tokenizer, resolution, trie):
    from byteify.alignment import align_text

    a = align_text(text, tokenizer, add_special_tokens=False)
    assert expand_byte_ids(a.byte_ids, trie) == brute_force_expand(a.byte_ids, resolution), text


def test_expand_reaches_past_the_owning_token(tokenizer, resolution, trie):
    """The expansion must not stop at the owning token's first byte."""
    from byteify.alignment import align_text

    text = "unbelievable the flowerbed antidisestablishmentarianism"
    a = align_text(text, tokenizer, add_special_tokens=False)
    owning = list(a.token_of_byte())
    expanded = expand_byte_ids(a.byte_ids, trie)

    differs = [i for i, (o, e) in enumerate(zip(owning, expanded)) if o != e]
    assert differs, "expected at least one byte whose expansion is not its own token"
    for i in differs:
        seq = list(resolution.token_bytes[expanded[i]])
        assert list(a.byte_ids[i - len(seq) + 1 : i + 1]) == seq
        assert expanded[i] != -1


def test_expand_is_a_suffix_match(tokenizer, resolution, trie):
    from byteify.alignment import align_text

    a = align_text("unbelievable", tokenizer, add_special_tokens=False)
    ids = expand_byte_ids(a.byte_ids, trie)
    for i, tid in enumerate(ids):
        seq = list(resolution.token_bytes[tid])
        assert list(a.byte_ids[i - len(seq) + 1 : i + 1]) == seq


def test_tie_break_prefers_higher_id(tokenizer, resolution, trie):
    """b'  ' is both a BPE token and an added token; the higher id must win."""
    from byteify.alignment import align_text

    a = align_text("  x", tokenizer, add_special_tokens=False)
    ids = expand_byte_ids(a.byte_ids, trie)
    assert ids[0] == next(t for t, b in enumerate(resolution.token_bytes) if b == b" ")
    assert ids[1] == max(t for t, b in enumerate(resolution.token_bytes) if b == b"  ")


def test_expand_handles_pseudo_bytes(tokenizer, trie):
    from byteify.alignment import align_text, discover_special_bytes

    a = align_text("hi", tokenizer, add_special_tokens=True)
    inverse = {pseudo: tid for tid, pseudo in discover_special_bytes(tokenizer).items()}
    ids = expand_byte_ids(a.byte_ids, trie, pseudo_byte_to_token=inverse)
    assert ids[0] == inverse[a.byte_ids[0]]
    assert ids[-1] == inverse[a.byte_ids[-1]]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expand_matches_brute_force_on_random_text(seed, tokenizer, resolution, trie):
    from byteify.alignment import align_text

    rng = random.Random(seed)
    alphabet = "abcdeABCDE .,!?_'0123456789éü日🙂"
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
    a = align_text(text, tokenizer, add_special_tokens=False)
    assert expand_byte_ids(a.byte_ids, trie) == brute_force_expand(a.byte_ids, resolution), repr(text)


def test_expand_is_fast_enough_for_training(tokenizer, trie):
    """A crude throughput guard: the trie must not become the training bottleneck."""
    from byteify.alignment import align_text

    text = "The quick brown fox jumps over the lazy dog. " * 60
    a = align_text(text, tokenizer, add_special_tokens=False)
    t0 = time.perf_counter()
    expand_byte_ids(a.byte_ids, trie)
    per_byte_us = (time.perf_counter() - t0) / a.n_bytes * 1e6
    assert per_byte_us < 20, f"{per_byte_us:.1f}us per byte over {a.n_bytes} bytes"