# byteify

Turn a subword **embedding model** into a **byte-native serving artifact**: a fixed-shape
ONNX graph that embeds text with no tokenizer in the path, described by a manifest that is
its only entry point.

```bash
uv sync --extra onnx
uv run byteify levels
uv run byteify build --model jhu-clsp/ettin-encoder-17m --level faithful \
    --data my-corpus.txt --out ./artifact
uv run byteify encode ./artifact --text "the quick brown fox"
```

Byteifying is Bolmo's method (Minixhofer et al., [*Nature*
2026](https://www.nature.com/articles/s41586-026-11111-4), [code](https://github.com/allenai/bolmo-core)),
ported from autoregressive language models to an encoder. Bolmo's method has no encoder
counterpart upstream — `bolmo-core` is decoder-only, mandates an LM head, and `olmo-core`
has no ModernBERT — so this reimplements the algorithm for the encoder setting rather than
switching a config.

## What it guarantees

Levels ascend **fit to the workload, not quality**. Every level starts from a Stage 1
checkpoint, so fidelity is constant across the ladder; what changes is how well the artifact
fits the data it was fitted for.

| level | name | recipe | guarantees |
| --- | --- | --- | --- |
| 1 | `faithful` | Stage 1 | boundary decisions and similarity structure match the source model, to the numbers in [`byteify/assets/thresholds.json`](src/byteify/assets/thresholds.json) |
| 2 | `adapted` | + contrastive on your corpus | everything level 1 guarantees, plus retrieval quality against any judgements you supply |
| 3 | `merged` | + task arithmetic from a post-trained variant | everything level 2 guarantees, plus that the delta is in the merged weights |

**Every level declares a guarantee and a gate that hard-fails.** If a check fails, the
command stops and **nothing is written** — an artifact that failed its own promise is worse
than no artifact, because it loads, it runs, and it is wrong.

Thresholds are **per source model**, published as data. Fidelity numbers do not transfer
between architectures, so pointing `byteify` at a model with no published floors is refused
in seconds rather than measured against someone else's numbers.

Stage 2 is **not** a level. It trains the global model away from the source model, which is
exactly what the fidelity constraint forbids. It stays a research path.

## The artifact

```
artifact/
  manifest.json        # the contract; the only entry point
  artifact.onnx        # the graph
  artifact.onnx.data   # external weights
  trie.json            # vocabulary byte strings, so inference needs no tokenizer
  encoder.pt           # encoder weights, for the Python path
```

`byteify encode` takes a directory and nothing else. No tokenizer is loaded and none is
needed: the byte stream comes from the rule the manifest declares, and the per-byte subword
suffix ids from the reverse trie the artifact ships, whose sha256 the manifest records so a
tampered one is refused.

The graph takes three inputs — `byte_ids`, `expanded_ids`, `byte_mask` — and returns one
normalised sentence embedding. The **byte axis** and the **patch budget** are fixed when the
artifact is built, so serving hardware can be sized deterministically. Text longer than the
declared byte axis fails loudly rather than truncating silently.

## What has and has not been established

**Established.** Byteification transfers to encoders. A local encoder **6.6% the size of the
source model** reproduces the source model's boundary decisions and its internal
representations, and its held-out similarity-matrix correlation and nearest-neighbour
agreement beat chance by two orders of magnitude. All on an M1 Max in minutes. The exported
graph matches eager PyTorch to ~1e-7, verified in CI and again by every build's gate.

**Not established.** That byteified models are *better*. Nothing in this repository can
detect a byte-level capability gain. The product therefore sells fidelity — which the
measurements support — and treats "byteified models are better" as out of scope rather than
as a marketing claim. See [`RESULTS.md`](RESULTS.md) §8, and issues #2 and #5 for the
evaluation that would be needed.

### A warning about embedding metrics

Raw cosine between mean-pooled embeddings reads **0.997** whether or not two models agree,
because pooled embeddings are strongly anisotropic — they share one dominant direction. An
earlier version of this harness reported 0.995 while the two models agreed on nearest
neighbours no better than chance. Everything here is reported **centred**, alongside
correlation of the two models' pairwise similarity matrices. Treat any byteified-encoder
result quoted as raw cosine as unverified.

## Reusability

Pointing byteify at a different embedding model is a configuration change. Verified across
seven models spanning two architectures and three tokenizer families:

**Alignment is lossless for all three families**, verified byte for byte against the training
path. That is a claim about alignment, not about quality — and it is not the same as being
buildable. Two source models have published fidelity floors, `jhu-clsp/ettin-encoder-17m` and
`hotchpotch/bekko-system-one-v0-17m`, so only those two can be built today; the other five are
**alignment-verified but refused by `byteify build`**, because fidelity numbers do not transfer
between architectures and borrowing another model's would be the same error as hardcoding them.

| model | arch | tokenizer | params | gated | buildable |
| --- | --- | --- | --- | --- | --- |
| `jhu-clsp/ettin-encoder-17m` | ModernBERT | byte-level BPE | 16.8M | yes | yes |
| `jhu-clsp/ettin-encoder-32m` | ModernBERT | byte-level BPE | 31.9M | no | no |
| `answerdotai/ModernBERT-base` | ModernBERT | byte-level BPE | 149M | no | no |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | WordPiece | 22.7M | no | no |
| `bert-base-uncased` | BERT | WordPiece | 110M | no | no |
| `xlm-roberta-base` | RoBERTa | SentencePiece | 278M | no | no |
| `hotchpotch/bekko-system-one-v0-17m` | ModernBERT | byte-level BPE | 16.8M | no | yes |

The last row is **not an embedding model**: it is a typed-decision model, and it ships as a
custom-code sentence-transformers repository rather than a plain checkpoint, so `AutoModel`
cannot open it. Both layouts resolve to the same encoder and tokenizer — see
[ADR 0012](docs/adr/0012-source-models-that-are-not-plain-checkpoints.md) — and
[`RESULTS.md`](RESULTS.md) §10.5 records what byteifying it cost, including a level-2 result
that came out negative.

To make one buildable, train it and publish its measured floors in
[`src/byteify/assets/thresholds.json`](src/byteify/assets/thresholds.json). It is data
precisely so that adding a model is a measurement rather than a decision.
`scripts/measure_thresholds.py` is that measurement.

Two caveats survive, and the manifest records both rather than hiding them:

- **Suffix-trie coverage varies sharply by family.** Complete for byte-level BPE; roughly
  one byte position in five uncovered for WordPiece, three in four on non-Latin script for
  SentencePiece. Level 1 is a materially weaker artifact on some families.
- **A tokenizer with a text normaliser is not tokenizer-free.** WordPiece and SentencePiece
  normalisers rewrite text before tokenising it, so reproducing the training stream means
  reproducing the normaliser. The manifest declares which rule an artifact was built for
  and `byte_stream.tokenizer_free` says whether the tokenizer-free claim holds for it.

## Running a byteified artifact

An artifact is a directory. `byteify encode` takes that directory and nothing else — no
config, no checkpoint path, no tokenizer. The source *model* is still fetched, because a
byteified encoder delegates its patch representations to the frozen encoder it was distilled
from; that delegation is what level 1's guarantee is.

```bash
uv sync --extra onnx
uv run byteify encode ./artifact --text "the quick brown fox"
uv run byteify encode ./artifact --file docs.txt --similarity   # blank line between paragraphs
uv run byteify encode ./artifact --onnx --text "the quick brown fox"   # the exported graph
uv run byteify inspect ./artifact                                # what it promised, and whether it held
```

Longer input fails rather than truncating, because an artifact's byte axis is fixed when it
is built and a serving path that quietly embedded a prefix would be worse than one that
refuses:

```
error: text needs 602 bytes but this artifact's byte axis is 512. It was built with
--max-bytes 512 and cannot take longer input; rebuild it with a larger axis rather than have
text silently truncated.
```

### The byteified `bekko-system-one-v0-17m`

A gated level-1 artifact for a typed-decision model, built and verified offline. `RESULTS.md`
§10.5 has the measurements and one negative result; the commands are:

```bash
# 1. the workload, rendered in the model's own prompt layout (public data, one time)
uv run --with pyarrow python scripts/prepare_decision_corpus.py \
    --train-texts 20000 --judgement-decisions 2000

# 2. the artifact: Stage 1, gates, graph export. ~8 minutes on an M1 Max.
HF_HUB_OFFLINE=1 uv run byteify build \
    --model hotchpotch/bekko-system-one-v0-17m --level faithful \
    --data data/corpus/open-jev-train.jsonl \
    --out artifacts/bekko-17m-faithful \
    --steps 6000 --batch-size 4 --max-bytes 512 --patch-budget 128

# 3. serve it, with no tokenizer in the path
HF_HUB_OFFLINE=1 uv run byteify encode artifacts/bekko-17m-faithful \
    --text "Candidate: billing: Charges, invoices, refunds"
```

```
byteify faithful: hotchpotch/bekko-system-one-v0-17m
  source layout: module (0_BekkoInference)
  6 checkpoint tensor(s) are not encoder weights and were not loaded: ['heads.choice.bias', …]
gates
  ok   byte_stream_lossless              0 <= 0
  ok   cos_emb_c                   0.91261 >= 0.8
  ok   sim_corr                    0.97981 >= 0.97
  ok   boundary_acc                0.99604 >= 0.99
  ok   trie_coverage                     1 >= 0.99
  onnx vs eager: max |diff| 1.37e-06
```

Two things to know before reading anything into it. Its **byte axis is 512 bytes**, and
Open-Jev decision queries have a median of 530 bytes — so half of them are truncated at
serving time, which is a property of the artifact, not of the model. And its tokenizer is
**byte-identical to the anchor's**, so `manifest.byte_stream.tokenizer_free` is `true`: no
tokenizer is needed, and no normaliser has to be reproduced.

Its three decision heads (`choice`, `noul`, `score`) are **not** in the artifact. It ships
the encoder and its pooled representation, which is what those heads read; grading it as a
decision model means attaching the source model's heads, which
`scripts/decision_harness.py` does:

```bash
uv run --with sentence-transformers --with pyarrow python scripts/decision_harness.py \
    --artifact artifacts/bekko-17m-faithful \
    --judgements data/judgements/open-jev-validation.jsonl --limit 150
```

```
kind       n  chance   source  src@axis  byteified   agree  meanTV
choice    37   0.323    0.622     0.595      0.270   0.216   0.259
noul      95   0.500    0.568     0.505      0.400   0.747   0.228
score     18   0.259    0.778     0.722      0.056   0.111   0.491
all      150   0.428    0.607     0.553      0.327   0.540   0.267
```

The harness runs the model's **own** decision path — cross-attention over a shared query
prefix, then per-task heads — and refuses to report byteified numbers unless its own source
path reproduces the model's `predict_groups` to within `--tolerance` (it does, to 1.5e-06).
`src@axis` is the truncation control: the same model on the bytes the artifact's axis can
hold. It costs 0.054, so the byteified column's 0.607 → 0.327 is byteification, not the
axis.

Reproducing the floors behind it:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/measure_thresholds.py \
    --model hotchpotch/bekko-system-one-v0-17m \
    --data data/corpus/open-jev-train.jsonl --checkpoint runs/stage1_bekko/stage1.pt \
    --judgements data/judgements/open-jev-validation.jsonl --out runs/thresholds_bekko.json
HF_HUB_OFFLINE=1 uv run python scripts/contrastive_tradeoff.py \
    --model hotchpotch/bekko-system-one-v0-17m --checkpoint runs/stage1_bekko/stage1.pt \
    --data data/corpus/open-jev-train.jsonl \
    --judgements data/judgements/open-jev-validation.jsonl
```

## Further reading

- **[`RESULTS.md`](RESULTS.md)** — findings, including the negative results and open problems.
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — how each part of the paper maps onto an encoder.
- **[`docs/adr/`](docs/adr/)** — the decisions, and what each one costs.
- **[`GLOSSARY.md`](GLOSSARY.md)** — patch, boundary indicator, stitching loss, level, gate, …

## Development

```bash
uv sync --extra dev --extra onnx
HF_HUB_OFFLINE=1 uv run pytest -q      # 299 tests, offline, ~2 min
uv run ruff check .
```

Research scripts live in `scripts/` and are not part of the documented surface; they run the
same code paths as the command. `scripts/prepare_corpus.py` fetches the corpus the
experiments use, and `scripts/prepare_decision_corpus.py` renders a typed-decision dataset
into reference texts and judgements in the model's own prompt layout.
`scripts/measure_thresholds.py` measures a source model's floors on the gate's own metric,
and `scripts/contrastive_tradeoff.py` maps the fidelity-for-retrieval trade that level 2
makes. Everything runs from the local HuggingFace cache; `HF_HUB_OFFLINE=1` is set
throughout.