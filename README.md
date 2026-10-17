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
six models spanning two architectures and three tokenizer families:

**Alignment is lossless for all three families**, verified byte for byte against the training
path. That is a claim about alignment, not about quality — and it is not the same as being
buildable. Only `jhu-clsp/ettin-encoder-17m` has published fidelity floors, so only it can be
built today; the other five are **alignment-verified but refused by `byteify build`**,
because fidelity numbers do not transfer between architectures and borrowing the anchor's
would be the same error as hardcoding them.

| model | arch | tokenizer | params | gated |
| --- | --- | --- | --- | --- |
| `jhu-clsp/ettin-encoder-17m` | ModernBERT | byte-level BPE | 16.8M | yes |
| `jhu-clsp/ettin-encoder-32m` | ModernBERT | byte-level BPE | 31.9M | no |
| `answerdotai/ModernBERT-base` | ModernBERT | byte-level BPE | 149M | no |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | WordPiece | 22.7M | no |
| `bert-base-uncased` | BERT | WordPiece | 110M | no |
| `xlm-roberta-base` | RoBERTa | SentencePiece | 278M | no |

To make one buildable, train it and publish its measured floors in
[`src/byteify/assets/thresholds.json`](src/byteify/assets/thresholds.json). It is data
precisely so that adding a model is a measurement rather than a decision.

Two caveats survive, and the manifest records both rather than hiding them:

- **Suffix-trie coverage varies sharply by family.** Complete for byte-level BPE; roughly
  one byte position in five uncovered for WordPiece, three in four on non-Latin script for
  SentencePiece. Level 1 is a materially weaker artifact on some families.
- **A tokenizer with a text normaliser is not tokenizer-free.** WordPiece and SentencePiece
  normalisers rewrite text before tokenising it, so reproducing the training stream means
  reproducing the normaliser. The manifest declares which rule an artifact was built for
  and `byte_stream.tokenizer_free` says whether the tokenizer-free claim holds for it.

## Further reading

- **[`RESULTS.md`](RESULTS.md)** — findings, including the negative results and open problems.
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — how each part of the paper maps onto an encoder.
- **[`docs/adr/`](docs/adr/)** — the decisions, and what each one costs.
- **[`GLOSSARY.md`](GLOSSARY.md)** — patch, boundary indicator, stitching loss, level, gate, …

## Development

```bash
uv sync --extra dev --extra onnx
HF_HUB_OFFLINE=1 uv run pytest -q      # 272 tests, offline, ~2 min
uv run ruff check .
```

Research scripts live in `scripts/` and are not part of the documented surface; they run the
same code paths as the command. `scripts/prepare_corpus.py` fetches the corpus the
experiments use. Everything runs from the local HuggingFace cache; `HF_HUB_OFFLINE=1` is set
throughout.