# Architecture

How Bolmo's byteification maps onto an *embedding* model, and why several parts of the
original method disappear, change, or have to be invented.

Source: Minixhofer et al., *Retrofitting language models to operate over bytes*,
*Nature* (2026), [doi:10.1038/s41586-026-11111-4](https://www.nature.com/articles/s41586-026-11111-4);
code at [allenai/bolmo-core](https://github.com/allenai/bolmo-core).

## The pipeline

```
text
  │  byteify.alignment         byte stream + B_subword (+ pseudo-bytes for specials)
  │  byteify.vocab             vocabulary as byte strings, per tokenizer family
  │  byteify.expand            T_SubwordSuffix ids per byte (reverse-order byte trie)
  ▼
ByteifiedEncoder
  │  ByteEmbedding             e_i = T_Byte(x_i) + T_SubwordSuffix(x_:i)
  │  LocalContextEncoder       bidirectional over bytes  (E)
  │  BoundaryPredictor         -> boundary logits        (B)
  │  pool_patches              bytes -> patches         (last byte of each patch)
  ▼
patch representations  (byteify.source.SourceEncoder)
  │  forward_layers(n)         the source model's frozen body, truncated at depth n
  ▼
Layer-n representations  →  stitching loss vs the source model's own layer-n states
  │
  ├─ depool + LocalContextEncoder (bidirectional)  →  byte-level MLM head  (Stage 2)
  └─ final_norm + mean pool                        →  sentence embedding
```

## Component-by-component

| Bolmo (decoder LM) | Here | Notes |
| --- | --- | --- |
| byte embedding (256) | kept | plus `n_special` pseudo-bytes |
| `T_SubwordSuffix` residual | kept | reverse-order byte trie, as in `olmo_core` |
| causal mLSTM local encoder, 1 layer | **bidirectional** sliding-window attention | nothing to decode incrementally |
| cosine boundary predictor, 1 byte lookahead | **full-context MLP** | no next byte to leak |
| last-byte-of-patch pooling | kept | no parameters, as in H-Net/Bolmo |
| global model | kept, frozen in Stage 1 | the source model's body, unmodified |
| depool: latest patch rep + projection | kept, **bidirectional** | adds the next patch as well in principle |
| causal mLSTM local decoder, 4 layers | **bidirectional**, optional | feeds byte-level MLM instead of an LM head |
| LM head over 512 entries (byte fused with boundary) | **256-entry masked-byte head** | no sampling, so no fusion needed |
| `L_B` boundary BCE, λ=4 | kept | reproduces the paper's >99% claim |
| `L_E` stitching at depth n=4 | kept, **n = full depth** | see below |
| `L_{D,Distill}` patch likelihood matching | **replaced** by `L_emb` | no output distribution exists |
| `L_CE` next-byte cross-entropy | **masked byte** cross-entropy | bidirectional model cannot predict forward |
| boundary merging (BPE / entropy / cross-entropy) | BPE per example | entropy variants need an auxiliary LM |
| task-arithmetic post-training | kept | transformer deltas only |

### Three simplifications the encoder setting buys

1. **Full-context boundary prediction.** Bolmo's headline architectural fix is a 1-byte
   lookahead, and the paper is explicit about the limit: subword tokenizers see the
   future, so they can decide `_Wor` is a token in `_Hello_Wor!` but not in
   `_Hello_World!`, while a causal predictor cannot. Bolmo must stay at 1 byte because
   anything more leaks the next byte during decoding. An embedder has no decode step, so
   the boundary predictor reads the whole sequence and the mismatch is closed outright
   rather than approximated.

2. **No output half.** The local decoder's purpose upstream is next-byte prediction.
   Here it is optional, and there is no LM head, no fused 512-entry vocabulary, no output
   boundary prediction, and no approximate-likelihood matching. That is roughly half of
   the original method's machinery.

3. **No up-projection.** Bolmo sets `d_local = d_global` because a projection bottlenecked
   representation rank, so we inherit that directly.

### Where the port has to invent

**The Stage 1 objective.** Bolmo's Stage 1 is minimised *exactly* when the byte model
emulates the subword model — they prove it. Without an output distribution there is no
such guarantee, so "recover the source model's behaviour" becomes a choice:

- `L_E` (the paper's stitching loss) reproduces *per-patch* behaviour. On its own it
  plateaus at similarity-matrix correlation ≈ 0.34.
- `L_emb` (batch-centred cosine on pooled embeddings) targets the quantity an embedding
  model is judged on. Adding it takes correlation to ≈ 0.88.

The two trade off against each other: raising `λ_emb` improves centred cosine while
degrading patch-level fidelity. The mixture is unresolved, and is the main open question
(`RESULTS.md`).

**A metric that must not be raw cosine.** Mean-pooled embeddings share one dominant
direction, so cosine between them sits near 1.0 regardless of content. An early version
of this harness reported 0.995 while the two models agreed on nearest neighbours at
chance. Every embedding-level claim is therefore reported centred, alongside similarity
matrix correlation.

## Reusability across models

Byteifying a different embedding model is a config change. Three places had to be made
tokenizer- and architecture-agnostic, and each turned up a real defect.

**Vocabulary → bytes** (`vocab.py`). Resolves byte-level BPE, WordPiece, SentencePiece
and byte-fallback vocabularies. Coverage is reported rather than assumed; entries that
cannot be resolved (e.g. `[UNK]`) are simply absent from the trie.

**Alignment** (`alignment.py`). The authoritative route depends on the family:

- Byte-level BPE → concatenate per-token byte strings. Offsets are **not** trustworthy:
  for `日本語のテキスト` the fast tokenizer reports overlapping duplicate spans
  `(2,3), (2,3)` for three consecutive tokens, so an offset-driven stream duplicates
  characters. (The byte stream is that of the NFC-normalised text, which is what the
  source model saw.)
- WordPiece / SentencePiece → offsets, with **gap absorption**. These pre-tokenizers drop
  whitespace, so token spans do not cover the text; the uncovered bytes are assigned to
  the following token, keeping a one-to-one partition of bytes to patches.

**Source encoder** (`source.py`). `HuggingFaceSourceEncoder` normalises ModernBERT
(`embeddings` + `layers` + fused `Wqkv` + two layer types) and the BERT family
(`embeddings` + `encoder.layer`) behind one interface exposing `d_model`, `n_layers`, the
embedding matrix, and `forward_layers(n)` for partial-depth forwards.

Verified with `scripts/verify_reuse.py` across six models: ettin-17m, ettin-32m,
ModernBERT-base (ModernBERT / byte-level BPE) and all-MiniLM-L6-v2, bert-base-uncased,
xlm-roberta-base (BERT / WordPiece / SentencePiece) — 100% vocabulary coverage and zero
alignment errors in all six.

## The stages beyond byteification

Byteification is an interface retrofit. Making the result *useful* needs two more stages, and
the paper provides neither, because it is not optimising retrieval.

**Contrastive** (`contrastive.py`). Symmetric InfoNCE with in-batch negatives and a learned
temperature, applied to pooled sentence embeddings. It runs on the same fixed-budget patch
axis used at inference, so training optimises the representation that will actually be
used. Positives are mined locally by nearest-neighbour in the *source* model's space, so no
external dataset is required and the positives are near-duplicates by the teacher's own
metric. The projection head is optional: with it the model follows SimCSE and retrieval is
evaluated on the projected vector, without it the model follows E5/BGE and evaluates on the
raw pooled embedding. This project measures fidelity on the raw embedding.

**Inference** (`inference.py`). Byte embedding, local encoder, boundary prediction, pooling,
the global model, output norm, mean pool. Three inputs and one output, no tokenizer.

### The patch budget

At inference the patch axis is fixed. This is forced by ONNX: training pools wherever the
boundary predictor fires, so the patch count is data-dependent, and ONNX graphs cannot have
data-dependent output shapes. The rule is to keep the most confident boundary positions up
to a budget and mask out any that fell below threshold — see
[ADR 0002](adr/0002-budgeted-patching-for-onnx.md).

The consequence worth carrying: **per-patch comparison against the token axis is only defined
while patches and tokens correspond one-to-one**. Above the source tokenizer's own
compression ratio they do not, and the compression sweep reports that metric as `NaN` rather
than comparing misaligned axes.

### What survives above the tokenizer's ratio

Merging boundaries makes patches longer, which is the mechanism behind the paper's
efficiency result and the reason byte-level models can be pushed to compression ratios a
subword model cannot reach without a larger vocabulary and a costlier softmax. But merging
is not free: it changes which bytes each patch represents, and Stage 1 was fitted to the
unmerged segmentation. Stage 2 is where the model adapts, which is why the paper keeps the
boundary loss in Stage 2 and why the sweep here is an upper bound on the cost.

## Cost, and what capacity is actually needed

The local encoder is small on purpose. The suffix residual retains the source model's
input embeddings, so the local encoder only has to supply the *contextual* part. Measured
held-out, 0.66M and 3.67M parameter local encoders score within noise of each other
(0.890 vs 0.853), with the larger one slightly worse from undertraining.

On the 16.8M ettin-encoder-17m anchor, the local encoder is 1.05M parameters — **6.6% of
the source model**. The suffix table is 12.89M parameters but is *reused* capacity,
initialised to the source model's own embedding matrix; the byte table is 67k.
## Where things live

```
src/byteify/
  alignment.py       byte<->subword alignment, B_subword, gap absorption
  vocab.py           vocabulary -> byte strings, per tokenizer family
  expand.py          T_SubwordSuffix via a reverse-order byte trie
  boundary_merge.py  per-example BPE boundary merging (compression experiments)
  source.py          generic source-encoder adapter + ByteifyConfig
  model.py           byte embeddings, local encoder, boundary predictor, pooling,
                     depooling, byte MLM head
  losses.py          L_B, L_E, L_emb, masked-byte CE, MLM masking
  stitching.py       batching, teacher states, metrics
  data.py            corpus loading, length-grouped batching
  contrastive.py     InfoNCE, learned temperature, pair mining, retrieval metrics
  task_arithmetic.py post-training transfer from a subword checkpoint
  inference.py       budgeted patching, ByteifiedEmbedder, ONNX export

scripts/
  prepare_corpus.py       materialise a corpus shard
  train_stage1.py         frozen global model, distillation
  train_stage2.py         end-to-end byte-level MLM
  train_contrastive.py    contrastive fine-tuning
  compression_sweep.py    the efficiency experiment
  fidelity_probe.py       single-batch Stage 1 diagnostic
  verify_reuse.py         reuse check across six models
  measure_compression.py  compression ratio by domain
  export_onnx.py          ONNX export + numerical verification

GLOSSARY.md     definitions; docs/adr/ decisions; RESULTS.md measurements
```

## Further reading

- [`GLOSSARY.md`](../GLOSSARY.md) — the vocabulary, with terms that mean something different
  here than in the paper marked **→ Here**.
- [`RESULTS.md`](../RESULTS.md) — measured numbers, including the negative results.
- [`adr/`](adr/) — decisions that were hard to reverse, surprising, and involved a real
  trade-off.
