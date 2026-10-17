# Byteifying an embedding model: findings

Port of Bolmo's byteification (Minixhofer et al., [*Nature* 2026,
doi:10.1038/s41586-026-11111-4](https://www.nature.com/articles/s41586-026-11111-4),
[code](https://github.com/allenai/bolmo-core)) from autoregressive LMs to an **embedding**
model, anchored on [`jhu-clsp/ettin-encoder-17m`](https://huggingface.co/jhu-clsp/ettin-encoder-17m).

Everything measured here was produced on an **M1 Max (64 GB, MPS)**, on the source model's
own domain. Design rationale is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); vocabulary
is in [`GLOSSARY.md`](GLOSSARY.md).

---

## 1. What had to be built, and why nothing could be reused

`allenai/bolmo-core` is decoder-only end to end. There is no encoder or embedding path to
switch on:

- `olmo-core` contains **no ModernBERT** anywhere, and `ettin-encoder-17m` *is* ModernBERT.
- `BolmoTransformer` **requires an `lm_head`**, and its local decoder exists only to produce
  next-byte logits.
- Boundary/byte fusion into a 512-entry vocabulary and output boundary prediction exist
  solely to make *generation* exact. An embedder never decodes.
- There is no contrastive/embedding train module.

So this is a port of the *algorithm*, not a configuration change.

## 2. Inventory: every part of the paper, and its status here

| Paper component | Status | Where |
| --- | --- | --- |
| Byte embedding | done | `model.ByteEmbedding` |
| `T_SubwordSuffix` residual | done, reverse-order byte trie | `expand.ByteTrie` |
| Non-causal boundary prediction | done, **strengthened to full context** | `model.MLPBoundaryPredictor` |
| Local encoder | done, **bidirectional** (mLSTM is CUDA/causal-only) | `model.LocalContextEncoder` |
| Last-byte-of-patch pooling | done | `model.pool_patches` |
| Global model = source encoder | done, generic adapter | `source.HuggingFaceSourceEncoder` |
| Depooling | done, bidirectional | `model.depool` |
| Local decoder | done, **bidirectional**, optional | `model.LocalContextEncoder` |
| LM head | **replaced** by a 256-way masked-byte head | `model.ByteMLMHead` |
| `L_B` boundary BCE (λ=4) | done | `losses.boundary_loss` |
| `L_E` stitching at depth n | done, **at full depth** | `losses.stitching_loss` |
| `L_{D,Distill}` likelihood matching | **invented replacement** | `losses.embedding_loss` |
| `L_CE` next-byte CE | **replaced** by masked-byte CE | `losses.masked_byte_loss` |
| Stage 1 (frozen global) | done | `scripts/train_stage1.py` |
| Stage 2 (end-to-end) | done | `scripts/train_stage2.py` |
| Contrastive fine-tuning | done | `scripts/train_contrastive.py` |
| Inference / embedding API | done | `byteify/inference.py` |
| ONNX export + numerical verification | done | `scripts/export_onnx.py` |
| Compression factor ↑ (BPE merges) | done | `boundary_merge.py`, `scripts/compression_sweep.py` |
| Entropy / cross-entropy merging | not implemented (paper finds both worse) | — |
| Task-arithmetic post-training | done | `task_arithmetic.py` |
| CUTE-style character data | not implemented | — |
| Task arithmetic *with real post-trained ckpt* | mechanism done, no suitable checkpoint | — |

## 3. Three simplifications the encoder setting buys

1. **Full-context boundary prediction.** Bolmo's headline architectural fix is a 1-byte
   lookahead, and the paper states the limit precisely: subword tokenizers see the future,
   so they can decide `_Wor` is a token in `_Hello_Wor!` but not in `_Hello_World!`, while a
   causal predictor cannot. Bolmo must stay at 1 byte because more would leak the next byte
   during decoding. **An embedder has no decode step**, so the predictor reads the whole
   sequence and the mismatch is closed outright rather than approximated.
2. **No output half.** No LM head, no fused 512-vocab, no output boundary prediction, no
   approximate-likelihood matching — roughly half the original method's machinery.
3. **No up-projection** (Bolmo's `d_local = d_global` finding carries over directly).

## 4. Results

### 4.1 Stage 1 fidelity — the central experiment

Configuration: **1.05M-parameter** local encoder (bidirectional sliding-window attention,
window 128, 1 layer, FFN 1024) = **6.6% of the 16.8M source model**. 20,000 training texts /
512 held-out from a 352 MB FineWeb-edu shard, batch 8 x 1024 bytes, full-depth stitching
(n = 7), `λ_B`=4, `λ_E`=1, `λ_emb`=1. All on MPS.

| Metric | random init | after Stage 1 |
| --- | --- | --- |
| boundary predictor accuracy | 0.000 | **0.990** |
| `cos` to source reps @ layer 7 | 0.136 | **0.899** |
| MSE @ layer 7 | 22.4 | 1.19 |
| raw cosine of pooled embeddings | 0.488 | 0.997 |
| **centred cosine of pooled embeddings** | −0.008 | **0.431** |
| **correlation of pairwise similarity matrices** | 0.145 | **0.340** |

*(That table is the single-objective run: `L_E` + `L_B` only, 1500 steps.)*

### 4.2 `L_emb` is what actually recovers the embedding geometry

Adding the encoder analogue of `L_{D,Distill}` — batch-centred cosine on pooled embeddings —
changes the picture completely:

| Stage 1 objective | `sim_corr` | `cos_emb_c` |
| --- | --- | --- |
| `L_E` + `L_B` only | 0.34 | 0.43 |
| `L_E` + `L_B` + `L_emb` | **0.89** | **0.83** |

So the stitching loss alone recovers *per-patch* behaviour and stalls; the discriminative
structure of the embedding space only appears once the objective mentions embeddings. This
is the single most important porting finding: **Bolmo's `L_E` is the wrong objective for an
embedding model**, because Bolmo is solved by a complementary output-side loss that an
embedder has no analogue of.

It is a genuine trade-off, though — raising `λ_emb` improves centred cosine while degrading
patch-level fidelity (`cos@7` 0.88 → 0.40 at `λ_emb`=1000). The right mixture is unresolved.

### 4.3 A measurement trap

Raw cosine between mean-pooled embeddings reads **0.997** and is close to worthless. Mean-pooled
embeddings are strongly anisotropic: nearly all of them point in nearly the same direction, so
cosine stays near 1.0 almost regardless of content. An early version of this harness reported
`cos_emb = 0.995` while the two models agreed on nearest neighbours at **chance**.

Every embedding-level claim here is therefore reported **centred** (batch mean removed), plus
correlation of the two models' full pairwise similarity matrices. Any byteified-encoder result
quoted as raw cosine should be treated as unverified.

### 4.4 Architecture decisions this harness settled

Bolmo describes Stage 1 as a vehicle for architecture search — "conduct stage 1 training to
quickly check whether a particular architecture has sufficient capacity". Each row is held-out
fidelity.

**Stitch at full depth, not n = 4.** The paper's n = 4 is a *cost* decision for a 7B model with
~32 layers. This encoder has 7 layers, so backpropagating through all of it is cheap, and the
target metric *is* the final representation.

| n | `cos@7` (20k texts) |
| --- | --- |
| 0 | 0.891 |
| 4 | 0.808 |
| 7 | **0.827** |

Training only to n = 4 leaves `cos@7` ≈ 0.61 — the last three layers never receive a matched
input, which is what Stage 2 is for.

**Local encoder capacity is not the bottleneck.** 0.66M and 3.67M parameter local encoders
score within noise (0.890 vs 0.853), the larger slightly *worse* from undertraining. Don't
spend capacity here; the retained subword embeddings do the work.

**Boundary prediction is easy.** 99.0% accuracy falls out of the suffix residual and is never
the bottleneck.

### 4.5 Efficiency: trading compression for fidelity

The paper's Fig. 2. Merging adjacent patch boundaries (per-example BPE, its best-performing
strategy) raises bytes/patch and shortens the global model's sequence. Measured on a Stage 1
checkpoint, merged boundaries applied at evaluation only:

| target | achieved B/patch | boundary acc | `sim_corr` | `cos_emb_c` | `nn_agree` |
| --- | --- | --- | --- | --- | --- |
| none | 4.62 | 0.941 | 0.792 | 0.783 | 0.583 |
| 4.3 | 4.70 | 0.939 | 0.781 | 0.781 | 0.625 |
| 6.0 | 6.00 | 0.912 | 0.592 | 0.675 | 0.375 |
| 8.0 | 7.95 | 0.885 | 0.521 | 0.606 | 0.250 |
| 12.0 | 11.93 | 0.853 | 0.394 | 0.515 | 0.229 |
| 16.0 | 15.79 | 0.840 | 0.419 | 0.495 | 0.250 |

**3.4x more compression for `sim_corr` 0.79 → 0.42**, with the smooth drop-off the paper
reports. This is an *upper bound on the cost*: the paper trains with merged supervision in
Stage 2, letting the model adapt; these numbers apply merged boundaries to a model that never
saw them. Note `cos@L` becomes undefined above the tokenizer's own ratio, because merging
breaks the one-to-one correspondence between patches and tokens that makes per-patch
comparison meaningful at all.

Also worth stating plainly: **byteification does not by itself improve compression.** With
`B_subword` supervision it inherits the source tokenizer's ratio. The gain in the paper comes
from *training* the boundary predictor towards merged boundaries.

### 4.6 What byteification inherits from the tokenizer

`scripts/measure_compression.py`:

| domain | bytes/patch | vs English |
| --- | --- | --- |
| english news | 5.87 | 1.31x |
| english prose | 4.48 | 1.00x |
| code (python) | 3.72 | 0.83x |
| russian | 3.34 | 0.75x |
| japanese | 3.24 | 0.72x |
| chinese | 2.54 | 0.57x |
| devanagari | 2.58 | 0.58x |
| dna | 2.04 | 0.46x |
| protein | 1.78 | 0.40x |

English lands at 4.48, essentially Bolmo's 4.3 target, so there is no compression penalty to
recover on English. The 3.3x spread is the quantitative form of the paper's argument: for
protein, DNA and non-Latin scripts the subword tokenizer needs ~2.3x more positions per byte
of text than English does.

## 5. Reusability

Byteifying a different embedding model is a **config change**. Three things had to be made
tokenizer- and architecture-agnostic, and each turned up a real defect.

`scripts/verify_reuse.py`, six models, **100% vocabulary coverage and zero alignment errors
in all six**:

| model | arch | tokenizer | params | local % | B/patch | gaps absorbed | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ettin-encoder-17m | modernbert | bytelevel | 16.8M | 2.7 | 3.30 | 0 | yes |
| ettin-encoder-32m | modernbert | bytelevel | 31.9M | 2.8 | 3.30 | 0 | yes |
| ModernBERT-base | modernbert | bytelevel | 149.0M | 2.0 | 3.30 | 0 | yes |
| all-MiniLM-L6-v2 | bert | wordpiece | 22.7M | 3.9 | 3.75 | 0 | yes |
| bert-base-uncased | — | wordpiece | — | — | 3.34 | 32 | yes |
| xlm-roberta-base | — | sentencepiece | — | — | 2.90 | 32 | yes |

**What this table does and does not say.** It says alignment is lossless across all three
tokenizer families. It does **not** say six gated, level-1 artifacts exist. `byteify` has
published fidelity floors for exactly one source model — `jhu-clsp/ettin-encoder-17m`, at
cos_emb_c 0.75, sim_corr 0.60, boundary_acc 0.90 — and refuses the other five with a
message saying so, because fidelity numbers do not transfer between architectures and
borrowing them would be the same error as hardcoding them. Turning a verified model into a
gateable one means training it and publishing its measured floors; see issue #15.

Three defects surfaced while doing this, each of which would have silently corrupted training:

1. **HF offsets are untrustworthy for byte-level BPE with multi-byte characters.** For
   `日本語のテキスト` the fast tokenizer reports overlapping duplicate spans `(2,3), (2,3)` for
   three consecutive tokens. Following the offsets **duplicates characters**. Byte-level BPE
   must be aligned by concatenating per-token byte strings; offsets are correct only for
   WordPiece/SentencePiece. There is now a test asserting exactly this.
2. **WordPiece and SentencePiece do not cover the text.** BERT's pre-tokenizer drops spaces,
   so in `"Hello World!"` the token `World` has span `(6,11)` and character 5 is covered by
   nothing. Concatenating token spans silently loses those bytes. They are now **absorbed**
   into the following token, keeping patches a one-to-one partition with tokens — which the
   stitching loss depends on.
3. **Uncased normalizers destroy information.** `bert-base-uncased` lowercases and strips
   accents, so `naïve` is tokenised `naive`. The token's byte string genuinely is not the
   source bytes, so the suffix trie falls back to a shorter match there. That is a property
   of the teacher, not a bug — but the suffix residual is weaker for such tokenizers.

## 6. Inference and ONNX export

### The inference path exists

`byteify/inference.py` turns text into embeddings and exports to ONNX. Three inputs
(`byte_ids`, `expanded_ids`, `byte_mask`), one output (a normalised sentence embedding),
no tokenizer at inference time -- `expanded_ids` is recovered from the byte stream with the
same reverse trie used in training, so the tokenizer is needed to build the vocabulary
once but never to serve.

### Budgeted patching: forced by the ONNX requirement

The patch axis is a **fixed** `n_patches`. This was not a design preference. Training pools
wherever the boundary predictor fires, so the patch count varies per input, and ONNX graphs
cannot have data-dependent shapes. The fix is to keep the `n_patches` highest-confidence
boundary positions (`TopK` + `Sort`, both ONNX operators) and mask out any that fell below
threshold.

It degrades gracefully in both directions, which is worth stating because it is a real
property rather than a workaround:

- Fewer confident boundaries than budget -> surplus slots are **masked out of pooling**, not
  fabricated. Verified by test.
- More confident boundaries than budget -> the least confident are dropped and their patches
  merge. Verified by test.

A side consequence: **per-patch comparison against the token axis is only defined while
patches and tokens correspond one-to-one**, which is why `cos@L` in the compression sweep is
reported as NaN above the tokenizer's own ratio.

### Budgeted patching costs real fidelity, and it is not free to ignore

The fidelity numbers elsewhere in this document — including §4.1's headline — are measured
under **oracle subword boundaries**. That is the right measurement for Stage 1's losses,
which pool with the true boundaries so the student's patch axis stays aligned with the
teacher's token axis. It is the *wrong* measurement for a serving artifact, which always
uses budgeted patching.

Both numbers are true. They are not the same claim.

Same checkpoint (6000 Stage 1 steps, 512-byte axis), only the patch budget varying:

| patch budget | implied bytes/patch | cos_emb_c | sim_corr | boundary_acc |
| --- | --- | --- | --- | --- |
| 64 | 8.0 | 0.585 | 0.441 | 0.9362 |
| **128** | **4.0** | **0.795** | **0.656** | **0.9362** |
| 192 | 2.7 | 0.783 | 0.630 | 0.9362 |
| 256 | 2.0 | 0.762 | 0.610 | 0.9362 |

(measured at 1500 steps; the shape is what matters here, not the absolute level)

Read across: the budget is not a free parameter. Too few merges patches and costs ~0.21
centred cosine; too many masks surplus slots and costs back some of it. `byteify` therefore
derives the budget from the byte axis at Bolmo's ~4 bytes per patch and **warns** when a
chosen budget does not fit.

At the fitted budget and a 6000-step run, the gate's own metric — budgeted patching, 128
patches, 512 bytes — reads:

| metric | oracle boundaries | budgeted (what the artifact does) |
| --- | --- | --- |
| `cos_emb_c` | 0.940 | **0.841** |
| `sim_corr` | 0.912 | **0.717** |
| `boundary_acc` | 0.944 | 0.944 |

Boundary accuracy is unchanged, because it does not depend on how patches are chosen — only
the pooled representation does. Those budgeted numbers, not the oracle ones, are what
`byteify/assets/thresholds.json` publishes, and they are why the thresholds had to be
re-measured rather than copied forward from §4.1.

### Verified numerically, not just exported

`scripts/export_onnx.py` exports, then re-runs the graph through ONNX Runtime and compares
against eager PyTorch:

| model | max abs diff | mean cosine |
| --- | --- | --- |
| ettin-encoder-17m, n_patches=24, 256 bytes | **3.7e-07** | **1.000000** |

An artifact that loads but computes something else is worthless, so this check is a test
(`tests/test_onnx_export.py`, 4 tests) rather than a printout.

### Six real bugs the export surfaced

Tracing this graph was the single most productive debugging session in the project; each
failure named the wrong thing, and each is now guarded by a test:

1. **`flatten(1, 2)` is not a substitute for `flatten(-1, d_model)`.** It merges heads into
   the sequence axis, yielding `[B, N * heads, head_dim]`. Only correct when `N == heads`,
   which is exactly the kind of bug that passes a shape check and silently reorders data.
2. **Unpacking a traced shape raises "too many values to unpack"** -- a traced `x.shape`
   carries an extra entry. Index by position instead.
3. **Re-reading `x.shape[i]` inside a generator expression** produces tensors of the wrong
   rank mid-trace (`[2, 1, 4, 64] is invalid for input of size 131072`).
4. **The TorchScript tracer bakes shapes in**, so `[B, N, D]` activations arrive at
   attention as 4-D and the errors name the wrong axis. Fixed by using the **dynamo**
   exporter (`dynamo=True`), which needs `onnxscript` and opset >= 18.
5. **Output name `embedding` collides** with the `nn.Embedding` initializer -- ONNX Runtime
   rejects the file with "Duplicate definition of name". The output is now
   `sentence_embedding`.
6. **Stacking `[1, n]` rows without squeezing** yields `[batch, 1, n]`, a 3-D tensor that
   propagates through the residual adds and surfaces as a 4-D activation inside
   `LocalAttention`. `export_onnx` now validates that all three inputs share a byte axis and
   fails at the call site with a clear message.

A related one, from the memory work: the local encoder's dense `[B, N, N]` band mask needs a
**concrete** sequence length, so the byte axis is fixed at export and callers pad to it.
`MAX_DENSE_ATTENTION_ELEMENTS` now raises with the real numbers instead of letting MPS fail
with "tensors requiring 64-bit indexing are not supported".

### Contrastive training

`scripts/train_contrastive.py` supplies the stage that was missing: Stage 1 recovers
behaviour, Stage 2 teaches byte-level competence, and **neither optimises retrieval**. This
trains symmetric InfoNCE with in-batch negatives on the budgeted patch axis, so there is no
train/inference mismatch. Positives are mined locally with the *source* model
(`contrastive.mine_pairs`), so no external dataset is needed.

21 tests cover the objective: it is zero for a single example (no negatives), matches a
manual computation, is symmetric, penalises wrong pairs, and uses a learned temperature
clamped away from collapse. Retrieval metrics are reported as recall/MRR/nDCG rather than
cosine, for the anisotropy reason in section 4.3.

Not yet run at scale: 60 steps on 300 mined pairs is a smoke test, and `nn_agree` there is
noise at that budget.

## 7. Things checked that are not concerns

- **Alignment is lossless** for byte-level BPE: all 50,368 vocabulary entries map to byte
  strings, and concatenating per-token bytes reproduces the tokenized text on 19 samples
  spanning CJK, emoji, RTL, tabs, and long identifiers.
- **NFC normalization** means the byte stream is the bytes of the *normalized* text. This is
  intentional — it is what the source model saw.
- **Byte-level BPE never splits a multi-byte character**, so per-token byte strings are
  independently valid UTF-8 and alignment cannot desynchronize.
- **The suffix residual genuinely reaches past the owning token.** The longest vocabulary
  suffix of `b"Hello"` is `"Hello"` even when the tokenizer emits `["H", "ello"]`. Verified
  against a brute-force reference on random text.
- **Two vocabulary entries can share a byte string** (BPE `ĠĠ` vs the added token `"  "`).
  Ties resolve to the higher id, matching `olmo_core`'s last-insert-wins.
- **Data preprocessing is not a bottleneck.** Memoising the vocabulary resolution took batch
  encoding from 1003 ms to 19 ms (53x); the trie itself costs 0.6 µs/byte.

## 8. Honest status and what is left

**Validated.** The byteification machinery transfers to encoders. A local encoder 6.6% the size
of the source model reproduces the source model's internal representations (cos 0.90 at full
depth) and its boundary decisions (99.0%), both consistent with the paper's claims. With
`L_emb`, held-out similarity-matrix correlation reaches 0.89 and nearest-neighbour agreement
0.31 against a chance rate of 0.008.

**Not established.** This is all *fidelity to the source model*. The paper's actual claim is
that byteified models are **better** — better character understanding, better on code and
scientific data, at competitive speed. None of that is demonstrated yet: it requires Stage 2
training and retrieval/character-level evaluation against the baselines in §9.

**Open problems, in the order I would attack them:**

1. **The `L_E` / `L_emb` trade-off** is unresolved and appears to be the central porting
   question. Candidates: a whitened stitching loss; matching similarity matrices directly
   rather than vectors; or a mixture scheduled over training.
2. **Stage 2 is implemented but untrained.** Expect fidelity to the source model to *drop*
   while byte-level competence rises; that trade is the point of the stage and needs
   evaluation that can see the gain, not just the loss.
3. **No evaluation that measures the actual claim.** `contrastive.retrieval_metrics` computes
   recall/MRR/nDCG, but there is no retrieval dataset wired up and nothing has been run at a
   budget where the number means anything. Non-English / code / bio are where byte level
   should win (see the compression table).
4. **Contrastive training is a smoke test only** — 60 steps on 300 mined pairs. The loss
   behaves correctly in tests but has never been trained to a useful retrieval accuracy.
5. **Character-level (CUTE-style) data** from the paper is not implemented; the paper adds
   75M tokens of it specifically to accelerate character understanding.
6. **Task arithmetic is verified mechanically but not semantically** — no post-trained ettin
   checkpoint exists to merge. The end-to-end test substitutes a synthetic delta and confirms
   representations move toward the "post-trained" model.
7. **Scale.** 17M is the smallest anchor in the suite. Whether the local encoder stays at 6.6%
   of the source as d_model grows is untested; `verify_reuse` shows the plumbing works at 149M
   but nothing has been trained.
8. **The exported ONNX graph fixes the byte axis** at `--max-bytes`. That is acceptable for a
   padded serving path but means variable-length inputs must be padded to one size, which is
   a real constraint for long documents and worth revisiting if byte level is used on them.

## 9. Baselines the claim has to beat

Byte-level encoders have prior art — Gecko/ModalityLM is a byte-based encoder trained from
scratch — so the comparison has to be:

- the source model fine-tuned as an embedding model (`ettin-encoder-17m` + sentence-transformers),
- a byte-level ModernBERT encoder trained from scratch at the same size and budget,
- the byteified model.

And the metrics must be anisotropy-aware: similarity-matrix correlation on retrieval, not raw
cosine.

## 10. Resource profile

Everything runs on an M1 Max (64 GB) under macOS, where CPU and GPU share one memory pool
and MPS's caching allocator returns memory only on an explicit `torch.mps.empty_cache()`.
An earlier long run at batch 8 x 1024 bytes grew the allocator until the machine was
**7.0 GB into 8.0 GB of swap** and throughput collapsed. The trainers were changed:

- defaults lowered to **batch 4 x 512 bytes**;
- teacher states computed at **only the depth needed** (`SourceEncoder.representation_at`)
  instead of materialising every hidden state;
- `torch.mps.empty_cache()` every 100 steps on MPS;
- `verify_reuse.py` collects each model before loading the next.

Measured after the change:

| entry point | peak RSS | wall clock |
| --- | --- | --- |
| `train_stage1.py`, 200 steps + 2 evals | **0.84 GB** | 24 s |
| `verify_reuse.py`, 6 models | **0.79 GB** | ~2 min |

Swap no longer grows during a run. A higher-throughput configuration is still available via
`--batch-size` / `--max-bytes`, at proportionally higher memory.

*(An attempt to cap the MPS allocator via `PYTORCH_MPS_HIGH_WATERMARK_RATIO` was reverted:
the ratio is version-dependent and this build rejects it outright.)*

## 11. Layout

```
src/byteify/alignment.py       byte<->subword alignment, B_subword, gap absorption
src/byteify/vocab.py           vocabulary -> byte strings, per tokenizer family
src/byteify/expand.py          T_SubwordSuffix via a reverse-order byte trie
src/byteify/source.py          generic source-encoder adapter + ByteifyConfig
src/byteify/model.py           byte embeddings, local encoder, boundary predictor, pooling,
                               depooling, byte MLM head, partial-depth forward
src/byteify/losses.py          L_B, L_E, L_emb, masked-byte CE, MLM masking
src/byteify/boundary_merge.py  per-example BPE boundary merging (compression experiments)
src/byteify/task_arithmetic.py post-training transfer from a subword checkpoint
src/byteify/data.py            corpus loading, length-grouped batching
src/byteify/inference.py       budgeted patching, ByteifiedEmbedder, ONNX export
src/byteify/contrastive.py     InfoNCE, learned temperature, pair mining, retrieval metrics
src/byteify/stitching.py       batching, teacher states, metrics

scripts/prepare_corpus.py      materialise a corpus shard (not run; corpora are local)
scripts/train_stage1.py        Stage 1: frozen global model, distillation
scripts/train_stage2.py        Stage 2: end-to-end byte-level MLM
scripts/compression_sweep.py   the efficiency experiment
scripts/fidelity_probe.py      single-batch Stage 1 diagnostic
scripts/verify_reuse.py        reuse check across six models
scripts/measure_compression.py compression ratio by domain
scripts/train_contrastive.py  contrastive fine-tuning (the retrieval objective)
scripts/export_onnx.py         ONNX export + numerical verification against eager
```

## 12. Reproducing

```bash
uv sync --extra dev
HF_HUB_OFFLINE=1 uv run pytest -q
uv run python scripts/verify_reuse.py
uv run python scripts/measure_compression.py
uv run python scripts/train_stage1.py --steps 2000 --n-train 4000 --out runs/stage1
uv run python scripts/compression_sweep.py --init runs/stage1/stage1.pt
uv run python scripts/train_stage2.py --init runs/stage1/stage1.pt --steps 2000
uv run python scripts/train_contrastive.py --init runs/stage1/stage1.pt --steps 500
uv run scripts/export_onnx.py --init runs/contrastive/contrastive.pt
```

`HF_HUB_OFFLINE=1` is set throughout: everything runs from the local cache and the local corpus.