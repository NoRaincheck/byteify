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

`scripts/verify_reuse.py`, seven models, **100% vocabulary coverage and zero alignment errors
in all seven**:

| model | arch | tokenizer | params | local % | B/patch | gaps absorbed | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ettin-encoder-17m | modernbert | bytelevel | 16.8M | 2.7 | 3.30 | 0 | yes |
| ettin-encoder-32m | modernbert | bytelevel | 31.9M | 2.8 | 3.30 | 0 | yes |
| ModernBERT-base | modernbert | bytelevel | 149.0M | 2.0 | 3.30 | 0 | yes |
| all-MiniLM-L6-v2 | bert | wordpiece | 22.7M | 3.9 | 3.61 | 0 | yes |
| bert-base-uncased | — | wordpiece | — | — | 3.51 | 68 | yes |
| xlm-roberta-base | — | sentencepiece | — | — | 3.04 | 46 | yes |
| bekko-system-one-v0-17m | modernbert | bytelevel | 16.8M | 2.7 | 3.30 | 0 | yes |

The last row resolves through a custom-code module layout rather than a plain checkpoint
(§10.5, ADR 0012), and it is not an embedding model: it is a typed-decision model, byteified
on its own workload.

**What this table does and does not say.** It says alignment is lossless across all three
tokenizer families. It does **not** say seven gated, level-1 artifacts exist. `byteify` has
published fidelity floors for two source models — `jhu-clsp/ettin-encoder-17m`, at
cos_emb_c 0.75, sim_corr 0.60, boundary_acc 0.90, and `hotchpotch/bekko-system-one-v0-17m`
at 0.80 / 0.97 / 0.99 — and refuses the other five with a message saying so, because
fidelity numbers do not transfer between architectures and borrowing them would be the same
error as hardcoding them. Turning a verified model into a gateable one means training it and
publishing its measured floors; see issue #15, and §10.5 for what that cost on the second
model.

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
3. **Level 2 has no retrieval objective that fits a decision workload.** A retrieval
   judgement set now exists (2,000 held-out Open-Jev decisions, §10.5) and the gate is
   wired, but contrastive training on mined near-duplicate pairs does not improve candidate
   ranking on it — seven rungs measured, and the only one that both held level 1's floors and
   beat 0.45 recall@1 did no contrastive training at all. Non-English / code / bio are still
   where byte level should win (see the compression table). What is needed is positives mined
   from query-to-candidate pairs with the judgement set split so the gate grades on unseen
   decisions.
4. **Contrastive training is a smoke test only** — it has never been trained to a useful
   retrieval accuracy, and on the one real judgement set available it does not get there.
   The four defects that had to be fixed to run it at all are in §10.5; the objective is the
   remaining problem, not the plumbing.
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

## 10.5 A second source model, and what it cost to make it gateable

`jhu-clsp/ettin-encoder-17m` is a plain HuggingFace checkpoint, so pointing `byteify` at
another embedding model really was a configuration change — until
[`hotchpotch/bekko-system-one-v0-17m`](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m),
which is not. It is a typed-decision model: instructions, application state and candidate
definitions in, a probability distribution over those candidates out. It is also a
custom-code sentence-transformers repository, so `AutoModel` cannot read it at all:

```
modules.json                              -> the module directory, 0_BekkoInference
0_BekkoInference/backbone_config.json     -> the config is not called config.json
0_BekkoInference/model.safetensors        -> keys prefixed `encoder.backbone.`
0_BekkoInference/heads.{choice,noul,score}-> three task heads, beside the encoder
0_BekkoInference/tokenizer/               -> the tokenizer is not at the root
```

This mattered more than it looks. A byteified encoder delegates its patch representations to
the frozen source encoder — level 1's fidelity guarantee *is* that delegation — so the source
model is re-fetched on every `byteify encode` call. A repository byteify can build from but
cannot serve is not a supported target; it would have produced an artifact that trained,
gated, exported, and then failed on first use. [ADR 0012](docs/adr/0012-source-models-that-are-not-plain-checkpoints)
resolves both layouts to the same model and tokenizer and names which one it found. Plain
checkpoints still go through `AutoModel`, so the ordinary path is unchanged.

Two details in that work are worth recording because both were silent:

* `AutoTokenizer.from_pretrained` does not raise `OSError` for a repository whose layout it
  does not recognise. A directory that exists and holds no `config.json` is a `ValueError`
  naming the missing key. Catching only `OSError` meant the fallback never ran, and the
  failure surfaced as "unrecognised model" from deep inside `transformers`.
* The task heads are stored beside the encoder under a prefix no encoder parameter claims.
  A remap that discarded them quietly would be indistinguishable from one that found the
  wrong prefix, so they are stripped from a **declared** prefix list and reported by name at
  build time: six tensors, `heads.{choice,noul,score}.{weight,bias}`.

### The model is byte-identical to the anchor's tokenizer

| | anchor | bekko |
| --- | --- | --- |
| architecture | ModernBERT, 7 × 256, 4 heads | identical |
| vocabulary | 50,368 byte-level BPE entries | identical, entry for entry |
| special-token pseudo-bytes | 7 | identical |
| suffix-trie coverage | 1.00 | 1.00 |

So none of the tokenizer-family caveats in §5.6 apply here: byte-level BPE with no
normaliser, coverage complete, `tokenizer_free: true`. What differs is the workload, not the
bytes.

### Reference data has to be the model's own workload

Training the byte interface on Project Gutenberg and gating it on decisions would measure a
model for a workload it never sees. `scripts/prepare_decision_corpus.py` renders
[`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) the way
`bekko-system-one` renders it at inference time — `Instruction: …\nState: …` in the
`instruction_state` prefix layout, `Candidate: {id}: {description}` per candidate, with a
Noul decision's two authored meanings folded into the state JSON exactly as `input_groups`
composes them. 20,000 texts from the train split; 2,000 held-out decisions from validation
(the split policy is by `group_id`, so no decision shares a state with another split).

### Measured floors

`scripts/measure_thresholds.py` measures on the gate's own path — budgeted patching, 512-byte
axis, patch budget 128, batch 4, the gate's 64 held-out texts — and reports the random-init
contrast, because a floor is only meaningful next to the number an untrained interface gets:

| metric | random init | after Stage 1 (6,000 steps) | published floor |
| --- | --- | --- | --- |
| centred cosine `cos_emb_c` | 0.078 | **0.889** | 0.80 |
| similarity-matrix correlation `sim_corr` | 0.949 | **0.985** | 0.97 |
| boundary accuracy | 0.671 | **0.996** | 0.99 |
| suffix-trie coverage | — | 1.000 | 0.99 |
| raw cosine | 0.164 | 0.989 | — |

`byteify build --level faithful` then cleared all five checks (cos_emb_c 0.913, sim_corr
0.980, boundary 0.996, trie coverage 1.00) and the exported graph matched eager PyTorch to
1.4e-06. The artifact then serves through its manifest alone on both the Python and the ONNX
path, with no tokenizer present.

**`sim_corr` barely discriminates on this corpus, and saying so matters.** An untrained byte
interface already reaches 0.949 against a trained 0.985. Rendered decisions from one
generator are homogeneous — same instruction templates, same state shapes — so almost any
embedding reproduces their coarse cluster structure, and the centred cosine is doing almost
all of the separating. The floor is set at 0.97 because it can be set at all; the numbers
that make this gate able to fail are `cos_emb_c` (0.078 → 0.889) and `boundary_acc` (0.671 →
0.996). This is the anisotropy warning of §4.3 arriving from the opposite direction: a
metric that reads 0.95 on an untrained model is not measuring what its name suggests.

### Two defects the measurement found, both of which were in the gate

**The gate graded sixteen texts.** The same Stage 1 checkpoint measured centred cosine 0.662
through the gate on one sixteen-text slice and 0.885 on another, from the same corpus. A
gate that size reports the slice it drew rather than the artifact. `GATE_TEXTS` is now 64,
which holds to about 0.01 and costs two seconds. Widening it can only help the anchor: its
published floors were measured on the smaller slice.

**Centring was applied per batch.** Removing the dominant direction means subtracting the mean
of the population, and a batch of four does not estimate a mean. The same checkpoint scored
0.885 through this repository's measurement path at batch 16 and 0.672 at batch 4 — on
identical texts. `measure_fidelity` now pools every batch and centres once. The anchor's
floors were re-measured under the corrected metric and are unchanged (0.824 / 0.629 / 0.949
against floors 0.75 / 0.60 / 0.90).

A third defect fell out of the same change: `cos_emb_raw` was computed from the already
centred vectors, so it was *identical to* `cos_emb_c` by construction — the raw number ADR
0005 exists to keep visible was not being kept visible. It now reads 0.989 against a centred
0.889, which is the anisotropy, visible.

### Level 2 could not be built, and the reason is not a bug

Level 2 declares that it guarantees everything `faithful` guarantees **plus** retrieval
quality on the supplied judgements. Measured on 2,000 held-out Open-Jev decisions:

| | recall@1 | MRR |
| --- | --- | --- |
| chance (mean 1/k over each decision's own pool) | 0.419 | — |
| **the source model itself** | **0.262** | 0.587 |
| byteified, three Stage 1 runs of one recipe | 0.488 / 0.301 / 0.281 | 0.697 / 0.612 / — |
| after the level-2 contrastive stage | 0.551 | 0.732 |

Two facts fall out, and the second is the one that decided it.

**The source model is below chance at this.** Ranking candidates by cosine between mean-pooled
embeddings is not how `bekko-system-one` decides; it decides through its three heads over a
shared-prefix encoder. So "retrieval quality on this workload", measured this way, is not a
capability of the model, and a byteified artifact beating its own source on it (0.488 vs
0.262) says the metric is measuring something else — not that the artifact is better.

**The spread between identical runs is wider than any margin.** Three Stage 1 runs of one
recipe scored 0.488, 0.301 and 0.281. A floor placed anywhere in that range fails half of
identical runs, which is a coin flip dressed as a gate. `scripts/contrastive_tradeoff.py`
walked seven rungs of the contrastive stage to see whether a gentler recipe could clear both
sides, and none could:

| contrastive steps | lr | `cos_emb_c` | `sim_corr` | boundary | recall@1 | level 1 holds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | — | 0.889 | 0.985 | 0.996 | 0.488 | yes |
| 150 | 2e-5 | 0.847 | 0.985 | 0.996 | 0.295 | yes |
| 500 | 2e-5 | 0.806 | 0.980 | 0.995 | 0.295 | yes |
| 1500 | 2e-5 | 0.772 | 0.987 | 0.991 | 0.299 | no |
| 150 | 1e-4 | 0.595 | 0.949 | 0.967 | 0.291 | no |
| 500 | 1e-4 | 0.543 | 0.930 | 0.914 | 0.273 | no |
| 1500 | 1e-4 | 0.573 | 0.947 | 0.867 | 0.524 | no |

The only rung that both holds level 1 and beats 0.45 is the one that does no contrastive
training at all. A second experiment — freezing `embed.byte` and `embed.suffix`, the tables
initialised from the source model's own embedding matrix, and training only the local
encoder and the boundary head — preserved fidelity (cos_emb_c 0.869 after 1,500 steps) and
left recall@1 at 0.287. So the trade-off is not a matter of training too hard or touching
the wrong parameters: contrastive training on mined near-duplicate pairs does not improve
candidate ranking for this workload at all.

The consequence is a refusal, which is the product behaving correctly. `byteify build
--level adapted --judgements …` for this model stops in seconds:

```
error: hotchpotch/bekko-system-one-v0-17m has no published retrieval floor, and judgements
were supplied.
Measure retrieval for this source model and add `retrieval_recall_at_1` to
byteify/assets/thresholds.json.
```

**So this model publishes fidelity floors and no retrieval floor**, and
`require_retrieval_floor` makes a level-2 build with judgements impossible until someone
measures a floor that the run-to-run spread can support. What would actually be needed is a
different objective for level 2 on a decision workload — mining positives from
query-to-gold-candidate pairs rather than from text near-duplicates — with the judgement set
split so the gate grades on decisions the contrastive stage never saw. That is a design
change with a leakage hazard attached, and it is not something to smuggle in as a recipe
tweak.

### Four bugs the level-2 path had, none of which any test reached

`--level adapted` had never run end to end. Reaching it took:

1. `train_contrastive_on_corpus` indexed the `ByteBatch` that `encode_texts` returns as
   though it were a sequence of rows — `TypeError` on the first step.
2. `infonce` returns a `ContrastiveLoss` record, and the caller called `.backward()` on it.
   The research script that shares the loss already used `.total`.
3. `ContrastiveHead.forward` returns `(projection, temperature)`. The caller passed the tuple
   to `infonce`, so the learned temperature — the module's one distinct idea — never trained.
4. `_budgeted_patch_reps` asked for `embedder.n_patches` patches regardless of how many byte
   positions the batch had, and died on any corpus of short texts including this repository's
   own synthetic fixture. The serving path already clamped this; the training path did not.

Each was one line from the interface as documented, and each was invisible because the
unit-testable parts of the module were tested and the one level that calls them was not. The
regression test is four steps on the synthetic fixture (`test_contrastive_training_on_a_corpus_runs_and_lowers_its_own_loss`).

### Grading the artifact in the decision path, not beside it

The retrieval numbers above are a proxy, and §10.5's own caveat says so: the source model is
below chance at them because ranking candidates by cosine between mean-pooled embeddings is
not how it decides. Reading `inference_v0.py` gives the real path —

```
prefix_ids  = tokenised query (instruction + state), CLS/SEP bracketed
doc_ids     = tokenised candidates ("Candidate: {id}: {description}")
hidden      = encoder(backbone)(prefix_ids, prefix_mask, doc_ids, doc_mask, owners)
              -- the prefix is encoded alone; every candidate's tokens attend to
                 [its query's keys, its own keys]
pooled      = masked mean over the candidate's token positions
scores      = heads[task](pooled)            -- one scalar per candidate
probs       = softmax(scores) within the decision
```

— a cross-attention encoder with three task heads, not an embedding. `scripts/decision_harness.py`
runs that path for the source model and for the artifact, substituting only the
representation: the artifact's pooled vector for the query and for each candidate enters as a
one-token sequence, and the backbone, the cross-attention and the heads are the source model's
own weights. The harness refuses to report byteified numbers unless its own source path
reproduces the model's `predict_groups`; it does, to 1.5e-06.

On 150 held-out validation decisions, 512-byte axis:

| kind | n | chance | source | source @ axis | byteified | agreement | mean TV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| choice | 37 | 0.323 | 0.622 | 0.595 | 0.270 | 0.216 | 0.259 |
| noul | 95 | 0.500 | 0.568 | 0.505 | 0.400 | 0.747 | 0.228 |
| score | 18 | 0.259 | 0.778 | 0.722 | 0.056 | 0.111 | 0.491 |
| **all** | 150 | 0.428 | **0.607** | 0.553 | **0.327** | 0.540 | 0.267 |

**Byteification, measured on the thing the model is for, costs about half the decision
accuracy**: 0.607 → 0.327, and the two paths pick the same candidate 54% of the time.
`source @ axis` is the control that makes this readable — the same model on the bytes the
artifact's axis can hold — and it costs 0.054, so the loss is byteification and not
truncation.

That is a harder result than the embedding-space gates report (centred cosine 0.913), and it
is not a contradiction. Averaging the representation over a whole document preserves a great
deal of what a pooled embedding is judged on while discarding exactly the per-token
contextualisation a cross-attention decision depends on: noul, whose two candidates differ by
one word, survives at 0.747 agreement; score, whose candidates are full rubric sentences,
collapses to 0.111. A byteified artifact should be expected to serve retrieval and
clustering tasks, and this model should be expected to lose its decision heads' accuracy
through it.

The honest scope of that number: it is 150 decisions, and the byteified path scores one pooled
vector where the source scores tokens attending to the query. It is the weakest task the
artifact can be asked to do, which is why it is the right one to ask.

### What was established

Byteifying a custom-code, non-embedding, decision model end to end: a gated level-1 artifact
that serves on both the Python and the ONNX path through its manifest alone, with no
tokenizer in the path, from a repository that `AutoModel` cannot open. What was not
established is anything about level 2 on this workload, and the numbers above say why rather
than leaving it to be discovered later.

### Two limits the artifact carries, measured

**The byte axis is the artifact's, not the workload's.** Open-Jev decision queries have a
median of 530 bytes and a 90th percentile of 1,750; at `--max-bytes 512` only 44% of them fit,
at 2,048 almost all do. A second artifact was built at a 2,048-byte axis with a 512-patch
budget, and **it failed its own gates** (sim_corr 0.812 against a floor of 0.97, boundary_acc
0.980 against 0.99), so it wrote nothing and no such artifact exists. The longer the byte
stream, the more of it one patch budget has to cover and the harder the fidelity constraint
is to hold — which is the same tension ADR 0009 records for levels, showing up here as an
axis length. The 512-byte artifact is the one that passes; its truncation cost is measured
above at 0.054 decision accuracy, not assumed to be zero.

**The decision heads are not in the artifact, and could not be.** Byteification distils the
encoder and its pooled representation. What the three task heads read from that
representation is what an artifact can serve; the heads themselves are the source model's
files, and shipping them would mean shipping a decision model rather than an embedding. The
harness attaches them to measure fidelity, and it measures a real loss.

## 11. Layout

```
src/byteify/alignment.py       byte<->subword alignment, B_subword, gap absorption
src/byteify/vocab.py           vocabulary -> byte strings, per tokenizer family
src/byteify/expand.py          T_SubwordSuffix via a reverse-order byte trie
src/byteify/source.py          generic source-encoder adapter + ByteifyConfig
src/byteify/hub.py             resolving a source model that is not a plain checkpoint
                               (plain `config.json`, or a sentence-transformers module dir)
src/byteify/model.py           byte embeddings, local encoder, boundary predictor, pooling,
                               depooling, byte MLM head, partial-depth forward
src/byteify/losses.py          L_B, L_E, L_emb, masked-byte CE, MLM masking
src/byteify/boundary_merge.py  per-example BPE boundary merging (compression experiments)
src/byteify/task_arithmetic.py post-training transfer from a subword checkpoint
src/byteify/data.py            corpus loading, length-grouped batching
src/byteify/inference.py       budgeted patching, ByteifiedEmbedder, ONNX export
src/byteify/contrastive.py     InfoNCE, learned temperature, pair mining, retrieval metrics
                               over judgement pools
src/byteify/stitching.py       batching, teacher states, metrics
src/byteify/bytestream.py      the declared byte-stream rule, one per tokenizer family
src/byteify/fidelity.py        one fidelity measurement, two explicit patch-selection rules
src/byteify/gates.py           gate checks, per-source-model thresholds as data
src/byteify/levels.py          level -> recipe + declared checks
src/byteify/manifest.py        the artifact contract; the only entry point
src/byteify/loader.py          manifest-only reconstruction
src/byteify/training.py        Stage 1, as a function rather than a script
src/byteify/build.py           `byteify build`: train, gate, export
src/byteify/cli.py             the command
src/byteify/assets/thresholds.json  published fidelity floors, per source model

scripts/prepare_corpus.py      materialise a corpus shard (not run; corpora are local)
scripts/train_stage1.py        Stage 1 as a run with a checkpoint at a chosen path
scripts/train_stage2.py        Stage 2: end-to-end byte-level MLM (research path)
scripts/compression_sweep.py   the efficiency experiment
scripts/fidelity_probe.py      single-batch Stage 1 diagnostic
scripts/verify_reuse.py        reuse check across seven models
scripts/measure_compression.py compression ratio by domain
scripts/train_contrastive.py  contrastive fine-tuning (the retrieval objective)
scripts/export_onnx.py         ONNX export + numerical verification, outside the command
scripts/prepare_decision_corpus.py  render a typed-decision dataset into reference texts
                                    and judgements, in the model's own prompt layout
scripts/measure_thresholds.py  measure one source model's floors on the gate's own metric,
                               against an untrained byte interface
scripts/contrastive_tradeoff.py  where level 2 stops being level 1: fidelity and retrieval
                                after each rung of the contrastive stage
scripts/decision_harness.py      grade an artifact inside the model's own decision path,
                                against the model's own predict_groups as the reference
```

## 12. Reproducing

The product is the command. Everything a practitioner needs is in `byteify --help` and
[the README](README.md); this section is for reproducing the *experiments* below.

```bash
uv sync --extra dev --extra onnx
HF_HUB_OFFLINE=1 uv run pytest -q          # the whole suite, offline, ~90s

# the product: a gated serving artifact
HF_HUB_OFFLINE=1 uv run byteify build \
    --model jhu-clsp/ettin-encoder-17m --level faithful \
    --data data/pg84.txt data/pg1342.txt data/pg2701.txt --out ./artifact \
    --steps 6000 --batch-size 4 --max-bytes 512 --patch-budget 128
HF_HUB_OFFLINE=1 uv run byteify encode ./artifact --text "the quick brown fox"

# the second source model (section 10.5): a custom-code module repository, byteified on
# its own workload. The corpus and the judgements are rendered from a public dataset.
uv run --with pyarrow python scripts/prepare_decision_corpus.py --train-texts 20000
HF_HUB_OFFLINE=1 uv run byteify build \
    --model hotchpotch/bekko-system-one-v0-17m --level faithful \
    --data data/corpus/open-jev-train.jsonl --out ./bekko-artifact \
    --steps 6000 --batch-size 4 --max-bytes 512 --patch-budget 128

# the measurements in this document
HF_HUB_OFFLINE=1 uv run byteify levels
HF_HUB_OFFLINE=1 uv run python scripts/verify_reuse.py        # section 5
HF_HUB_OFFLINE=1 uv run python scripts/measure_compression.py # bytes/patch by domain
HF_HUB_OFFLINE=1 uv run python scripts/compression_sweep.py --init <checkpoint.pt>

# the floors behind src/byteify/assets/thresholds.json (section 10.5)
HF_HUB_OFFLINE=1 uv run python scripts/measure_thresholds.py \
    --model hotchpotch/bekko-system-one-v0-17m \
    --data data/corpus/open-jev-train.jsonl --checkpoint runs/stage1_bekko/stage1.pt \
    --judgements data/judgements/open-jev-validation.jsonl --out runs/thresholds_bekko.json
HF_HUB_OFFLINE=1 uv run python scripts/contrastive_tradeoff.py \
    --model hotchpotch/bekko-system-one-v0-17m --checkpoint runs/stage1_bekko/stage1.pt \
    --data data/corpus/open-jev-train.jsonl \
    --judgements data/judgements/open-jev-validation.jsonl

# the decision path itself (section 10.5), with the model's own predict_groups as reference
uv run --with sentence-transformers --with pyarrow python scripts/decision_harness.py \
    --artifact artifacts/bekko-17m-faithful \
    --judgements data/judgements/open-jev-validation.jsonl --limit 150 \
    --out runs/decision_harness_bekko.json
```

`HF_HUB_OFFLINE=1` is set throughout: everything runs from the local cache and the local
corpus.

### The scripts are research scaffolding, not the interface

`scripts/` holds the experiments this document reports. Each one either calls the same code
path as the command or is a measurement the command does not make; none of them is part of
the documented surface, and `scripts/train_stage1.py` now delegates to
`byteify.training.train_stage1` so the two cannot drift into training different things.

Five of them were broken when this work started — three raised `NameError` because a
refactor moved symbols without updating the callers, and two failed at import on a removed
function. `ruff` had been reporting all of them as `F821`/`F401` for some time; nothing ran
it. They are fixed and `ruff check .` is clean.

Stage 2 and the contrastive trainer remain research paths. `scripts/train_stage2.py` is not
reachable from `byteify --level`, because Stage 2 trains the global model away from the
source model, which is what the fidelity guarantee forbids (ADR 0007, ADR 0009).