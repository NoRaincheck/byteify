"""Synthetic source models and tokenizers, built from nothing.

Every test that needs a source model builds one of these instead of downloading a real
checkpoint. They are deliberately tiny — two layers, a small hidden width, a few hundred
vocabulary entries — so a test can byteify one end to end in under a second, but they are
*real*: real HuggingFace architectures, real fast tokenizers that genuinely segment text,
persisted to disk and loaded back the way a published model would be.

Two things make these worth having over mocks. The tokenizer really produces byte offsets,
so the alignment bugs a mock would paper over are still reachable. And the model really
routes through the source-model interface, so a shape mismatch still fails here rather than
in production.
"""

from __future__ import annotations

import random

SYNTHETIC_FAMILIES = ("modernbert", "bert")
SYNTHETIC_TOKENIZER_FAMILIES = ("bytelevel", "wordpiece", "sentencepiece")

# Each architecture is paired with the tokenizer family it ships with in the wild: the
# anchor model is ModernBERT over byte-level BPE, the BERT family ships with WordPiece.
SYNTHETIC_MODEL_TOKENIZER = {"modernbert": "bytelevel", "bert": "wordpiece"}

_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_FFN = 64
_MAX_POSITIONS = 256

_WORDS = (
    "byte",
    "byteify",
    "patch",
    "boundary",
    "embedding",
    "encoder",
    "decoder",
    "retrofit",
    "suffix",
    "residual",
    "token",
    "tokenizer",
    "stream",
    "pseudo",
    "budget",
    "fidelity",
    "retrieval",
    "anisotropy",
    "cosine",
    "corpus",
    "gradient",
    "stitching",
    "alignment",
    "loss",
    "gate",
    "manifest",
    "artifact",
    "level",
    "recipe",
    "the",
    "a",
    "of",
    "and",
    "to",
    "in",
    "is",
    "it",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "be",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "no",
    "yes",
    "but",
    "or",
    "if",
    "then",
    "else",
    "when",
    "where",
    "which",
    "who",
    "what",
)

# Non-ASCII on purpose: the byte-level paths are exactly what these models get wrong.
_ODDITIES = (
    "café",
    "café",
    "naïve",
    "Ω",
    "日本語のテキスト",
    "한국어",
    "Привет",
    "مرحبا",
    "🎉",
    "👩‍👩‍👧‍👦",
    "ﬀ",
    "１２３",
    "—",
)


def synthetic_corpus(n: int = 600) -> list[str]:
    """Deterministic text with enough vocabulary diversity to train a small tokenizer."""
    rng = random.Random(0)
    out: list[str] = []
    for i in range(n):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(3, 12))]
        sentence = " ".join(words)
        if i % 7 == 0:
            sentence = f"{sentence} {rng.choice(_ODDITIES)}"
        out.append(sentence)
    return out


def build_tokenizer(family: str, corpus: list[str]):
    """A real fast tokenizer for one of the three families byteification supports."""
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    special = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    if family == "bytelevel":
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=special, show_progress=False)
        kwargs = {"unk_token": "[UNK]", "mask_token": "[MASK]"}
    elif family == "wordpiece":
        backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        backend.normalizer = normalizers.BertNormalizer(lowercase=True)
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(
            vocab_size=400, special_tokens=special, show_progress=False
        )
        kwargs = {"unk_token": "[UNK]", "mask_token": "[MASK]"}
    elif family == "sentencepiece":
        backend = Tokenizer(models.Unigram())
        backend.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always", split=True
        )
        backend.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=400, special_tokens=special, unk_token="[UNK]", show_progress=False
        )
        kwargs = {"unk_token": "[UNK]", "mask_token": "[MASK]"}
    else:
        raise ValueError(f"unknown tokenizer family {family!r}")

    backend.train_from_iterator(corpus, trainer)

    # A real tokenizer in every family byteify supports brackets its output with the class
    # and separator tokens. Without this the fixture emits no pseudo-bytes, which hides
    # exactly the behaviour the byte-stream rule has to reproduce: the encoder was trained
    # with pseudo-bytes at positions 0 and N-1 and has never seen the stream without them.
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1",
        special_tokens=[
            ("[CLS]", backend.token_to_id("[CLS]")),
            ("[SEP]", backend.token_to_id("[SEP]")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        bos_token="[CLS]",
        eos_token="[SEP]",
        **kwargs,
    )


def build_source_model(family: str, tokenizer):
    """A tiny but real source model, sized so a test can byteify it end to end."""
    vocab_size = len(tokenizer)

    if family == "modernbert":
        from transformers import ModernBertConfig, ModernBertModel

        config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_FFN,
            max_position_embeddings=_MAX_POSITIONS,
            pad_token_id=tokenizer.pad_token_id,
            layer_types=["full_attention"] * _LAYERS,
        )
        return ModernBertModel(config)

    if family == "bert":
        from transformers import BertConfig, BertModel

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_FFN,
            max_position_embeddings=_MAX_POSITIONS,
            pad_token_id=tokenizer.pad_token_id,
        )
        return BertModel(config)

    raise ValueError(f"unknown source family {family!r}")