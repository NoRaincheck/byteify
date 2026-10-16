"""Per-byte subword token expansion (``expanded_input_ids``).

Bolmo's local encoder does not start from a plain byte embedding. It adds, to every
byte position, the embedding of the longest vocabulary token whose byte string is a
*suffix* of the byte prefix ending at that position::

    e_i = T_Byte(x_i) + T_SubwordSuffix(x_:i)

This is what lets a single mLSTM layer stand in for the source model's whole input
embedding matrix: the pretrained subword embeddings are retained and simply broadcast
onto the bytes they cover. ``olmo_core`` calls the resulting per-byte id sequence
``expanded_input_ids`` and computes it with a byte trie.

The "longest suffix in the vocabulary" rule is genuinely not the same as "the token
that contains byte i". For ``"Hello"`` tokenised as ``["H", "ello"]`` the byte after
``o`` belongs to ``"ello"``, but the longest vocabulary suffix of ``b"Hello"`` is
``"Hello"`` -- a token that never appears in this input. Walking backwards past the
current token's start is therefore required, and is what
``olmo_core.data.tokenizer.ByteTokenizer.expand_byte_ids`` does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ByteTrie:
    """A byte-level trie over the vocabulary, mapping to token ids.

    Byte strings are stored **reversed**, so that walking a candidate suffix backwards
    (from its final byte towards its first) descends the trie one node per byte. This
    is what ``olmo_core.data.tokenizer.ByteTokenizer`` does, and it makes each lookup
    O(length of the longest matching suffix) rather than O(max_token_length^2).

    Nodes are stored as parallel lists of dicts keyed by node index, so no substring is
    materialised per step.
    """

    __slots__ = ("_children", "_max_depth", "_token_ids")

    def __init__(self, token_bytes: Mapping[int, bytes]) -> None:
        children: list[dict[int, int]] = [{}]
        token_ids: list[int | None] = [None]
        max_depth = 0

        for token_id, seq in token_bytes.items():
            if not seq:
                continue
            node = 0
            for depth, byte in enumerate(reversed(seq)):
                child = children[node].get(byte)
                if child is None:
                    child = len(children)
                    children[node][byte] = child
                    children.append({})
                    token_ids.append(None)
                node = child
                max_depth = max(max_depth, depth + 1)
            # A duplicate byte string keeps the higher id, matching tokenizers where
            # later additions win (this is observable: the BPE token "ĠĠ" and the
            # added token "  " both denote b"  ").
            if token_ids[node] is None or token_ids[node] < token_id:
                token_ids[node] = token_id

        self._children = children
        self._token_ids = token_ids
        self._max_depth = max_depth

    @property
    def max_depth(self) -> int:
        """Longest token in the vocabulary, in bytes."""
        return self._max_depth

    def __len__(self) -> int:
        return len(self._children)

    def longest_suffix_id(self, data: Sequence[int], end: int) -> int:
        """Id of the longest token that is a suffix of ``data[:end + 1]``.

        Returns ``-1`` when no vocabulary entry matches even the single byte at
        ``end``, which happens at pseudo-byte positions and for bytes that the source
        tokenizer's normalizer removed.
        """
        node = 0
        best = -1
        for i in range(end, max(-1, end - self._max_depth), -1):
            child = self._children[node].get(data[i])
            if child is None:
                break
            node = child
            token_id = self._token_ids[node]
            if token_id is not None:
                best = token_id
        return best

    def longest_suffix_id_batch(self, data: Sequence[int]) -> list[int]:
        """Vectorised :meth:`longest_suffix_id` over a whole byte stream."""
        return [self.longest_suffix_id(data, i) for i in range(len(data))]


def expand_byte_ids(
    byte_ids: Sequence[int],
    trie: ByteTrie,
    *,
    pseudo_byte_to_token: Mapping[int, int] | None = None,
) -> list[int]:
    """Per byte position, the id of the longest vocabulary token ending there.

    ``byte_ids`` may contain pseudo-bytes (>= 256) standing in for special tokens.
    ``pseudo_byte_to_token`` maps those back to their token ids so that a special
    token's position resolves to itself rather than failing the trie walk.
    """
    out: list[int] = []
    for i, byte in enumerate(byte_ids):
        if pseudo_byte_to_token is not None and byte >= 256:
            out.append(pseudo_byte_to_token[byte])
        else:
            out.append(trie.longest_suffix_id(byte_ids, i))
    return out