# byteify

Byteifying subword **embedding** models: porting Bolmo's byteification
(Minixhofer et al., [*Nature* 2026](https://www.nature.com/articles/s41586-026-11111-4),
[code](https://github.com/allenai/bolmo-core)) from autoregressive language models to an
encoder, anchored on [`jhu-clsp/ettin-encoder-17m`](https://huggingface.co/jhu-clsp/ettin-encoder-17m).

Bolmo's method has no encoder counterpart upstream — `bolmo-core` is decoder-only, mandates
an LM head, and `olmo-core` has no ModernBERT — so this reimplements the algorithm for the
encoder setting rather than switching a config.

## Read this first

- **[`RESULTS.md`](RESULTS.md)** — findings, including the negative results and open problems.
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — how each part of the paper maps onto an encoder.
- **[`GLOSSARY.md`](GLOSSARY.md)** — the vocabulary (patch, boundary indicator, stitching loss, …).

## Status

Validated: the machinery transfers. A local encoder **6.6% the size of the source model**
reproduces the source encoder's internal representations (cos 0.90 at full depth) and its
boundary decisions (**99.0%**), both consistent with the paper's claims. Held-out
similarity-matrix correlation reaches **0.89**, nearest-neighbour agreement 0.31 against a
chance rate of 0.008. All on an M1 Max in minutes.

You can also **embed text and export the model**: the ONNX graph matches eager PyTorch to
**3.7e-07** max absolute difference, cosine 1.000000, verified in CI.

Not established: that byteified models are *better*. That needs Stage 2 training and
retrieval/character-level evaluation against from-scratch byte encoders. See `RESULTS.md` §8.

## Inference and ONNX

```bash
uv sync --extra dev --extra onnx
uv run scripts/export_onnx.py --init runs/contrastive/contrastive.pt --max-bytes 512
```

Three inputs (`byte_ids`, `expanded_ids`, `byte_mask`) and one output (a normalised sentence
embedding). **No tokenizer is needed at inference** — the per-byte suffix ids are recovered
from the byte stream with the same reverse trie used in training.

The graph uses **budgeted patching**: a fixed number of patches, chosen as the
highest-confidence boundary positions, with low-confidence slots masked out of pooling. This
is forced by ONNX's static-shape requirement, and degrades gracefully — too few confident
boundaries means masked slots, too many means patches merge.

## Reusability

Pointing byteify at a different embedding model is a config change. Verified across six
models spanning two architectures and three tokenizer families, with 100% vocabulary coverage
and zero alignment errors:

| model | arch | tokenizer | params |
| --- | --- | --- | --- |
| `jhu-clsp/ettin-encoder-17m` | ModernBERT | byte-level BPE | 16.8M |
| `jhu-clsp/ettin-encoder-32m` | ModernBERT | byte-level BPE | 31.9M |
| `answerdotai/ModernBERT-base` | ModernBERT | byte-level BPE | 149M |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | WordPiece | 22.7M |
| `bert-base-uncased` | BERT | WordPiece | 110M |
| `xlm-roberta-base` | RoBERTa | SentencePiece | 278M |

## A warning about embedding metrics

Raw cosine between mean-pooled embeddings reads **0.997** whether or not two models agree,
because pooled embeddings are strongly anisotropic — they share one dominant direction. An
early version of this harness reported 0.995 while the two models agreed on nearest neighbours
no better than chance. Everything here is reported **centred**, alongside correlation of the
two models' pairwise similarity matrices. Treat any byteified-encoder result quoted as raw
cosine as unverified.

## Usage

```bash
uv sync --extra dev
HF_HUB_OFFLINE=1 uv run pytest -q

uv run python scripts/verify_reuse.py          # reuse check across six models
uv run python scripts/measure_compression.py   # bytes/patch by domain

# Stage 1: source model frozen, learn the byte interface (~0.85 GB RSS, 24 s / 200 steps)
uv run python scripts/train_stage1.py --steps 2000 --out runs/stage1

# The paper's efficiency experiment: compression vs fidelity
uv run python scripts/compression_sweep.py --init runs/stage1/stage1.pt

# Stage 2: end-to-end byte-level MLM
uv run python scripts/train_stage2.py --init runs/stage1/stage1.pt --steps 2000

# Contrastive fine-tuning: the retrieval objective Stage 1/2 do not optimise
uv run python scripts/train_contrastive.py --init runs/stage1/stage1.pt --steps 500

# Export to ONNX and verify it against eager PyTorch
uv run scripts/export_onnx.py --init runs/contrastive/contrastive.pt
```

Everything runs from the local cache and the local corpus; `HF_HUB_OFFLINE=1` is set
throughout.