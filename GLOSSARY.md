# Byteification Glossary

The language for retrofitting a subword **embedding model** so it operates on raw UTF-8
bytes, after the method described in Minixhofer et al., *Retrofitting language models to
operate over bytes* (*Nature* 2026). Definitions only: what a term *is*, not why we chose
it, which module implements it, or what it measured. Decisions live in `docs/adr/`, the
design in `docs/ARCHITECTURE.md`, the numbers in `RESULTS.md`.

Terms marked **→ Here** differ from the paper's usage. The paper describes a decoder-only
language model; this project ports the method to a bidirectional encoder, and a few terms
change meaning in the process.

## The retrofit

**Byteification**:
Converting a trained subword model into a byte-level model by replacing its input and
output interface while keeping its transformer body.
_Avoid_: retokenization, byte-ifying, de-tokenization

**Source model**:
The subword model being retrofitted.
_Avoid_: teacher model, parent model, donor model

**Global model**:
The source model's transformer body, which byteification leaves intact and which consumes
patch representations rather than tokens.
_Avoid_: backbone, encoder stack, trunk

**Local encoder**:
The shallow module that contextualises bytes before they are pooled.
→ **Here**: bidirectional. _Avoid_: tokenizer, byte encoder

**Local decoder**:
The module that returns patch representations to the byte axis, contextualised, for
byte-level prediction.
→ **Here**: bidirectional and optional; there is no autoregressive decoding to support.
_Avoid_: decoder (it decodes no tokens), upsampler

## Bytes, patches and boundaries

**Byte stream**:
The sequence of UTF-8 bytes a byteified model consumes, including pseudo-bytes standing in
for special tokens.
_Avoid_: raw bytes (the stream is normalized)

**Pseudo-byte**:
A value outside the byte range representing a special token, so the byte axis stays
aligned with the token axis.
_Avoid_: sentinel, mask token

**Patch**:
A contiguous run of bytes between two patch boundaries. Under oracle boundaries, exactly
one patch corresponds to each source token.
_Avoid_: chunk, segment, piece, token

**Boundary indicator** (`B_subword`):
The zero-one sequence marking the last byte of each patch. With oracle boundaries it is
the source tokenizer's segmentation projected onto bytes.
_Avoid_: mask (connotes attention)

**Boundary predictor**:
The module that scores where patches end.
→ **Here**: position-wise; all of its non-causality comes from the local encoder.
→ **In Bolmo**: cosine similarity against the *next* byte, because generation forbids
seeing further. _Avoid_: tokenizer, segmenter, patch head

**Oracle boundaries**:
Using `B_subword` itself as the boundaries instead of the predictor's output. Required in
Stage 1; forbidden in Stage 2.
_Avoid_: teacher forcing (close, but this is about which positions index patches)

**Compression ratio**:
Mean bytes per patch.
_Avoid_: compression factor (ambiguous about units)

**Patch budget**:
The fixed maximum number of patches a byteified encoder will emit.
_Avoid_: max_patches (an implementation name, not a concept)

**Budgeted patching**:
Spending a patch budget: choosing the most confident boundaries up to the budget and
masking out any that fell below threshold.
_Avoid_: truncation, top-k patching

**Gap**:
Bytes of the source text that no token covers, typically whitespace a pre-tokenizer drops.
_Avoid_: padding, slack

**Gap absorption**:
Assigning gap bytes to the following token, so patches still partition the byte stream
one-to-one with tokens.
_Avoid_: whitespace skipping

## The suffix residual

**Suffix residual** (`T_SubwordSuffix`):
The per-byte term added to each byte embedding: the embedding of the longest vocabulary
token whose bytes are a suffix of the byte prefix ending at that byte.
_Avoid_: expanded embeddings, context embedding

**Suffix expansion**:
The per-byte sequence of those vocabulary ids.
_Avoid_: expanded input ids, context ids

## The recipe

**Stage 1**:
Training the new byte interface while the global model stays frozen, so the byteified
model recovers the source model's behaviour.
_Avoid_: pretraining

**Stage 2**:
Training everything so the model learns to use byte-level information rather than imitate
the subword model.
_Avoid_: finetuning, annealing

**Stitching loss** (`L_E`):
Matching the student's and the source model's representations at depth `n` *inside* the
frozen global model, rather than at its input.
_Avoid_: distillation loss, embedding loss

**Embedding alignment loss** (`L_emb`):
Matching the two models' pooled sentence embeddings.
→ **Here**: invented. The paper's counterpart, `L_{D,Distill}`, serves the same purpose
but matches next-token likelihood. _Avoid_: contrastive loss (no negatives are involved)

**Boundary loss** (`L_B`):
Binary cross-entropy teaching the predictor to reproduce `B_subword`.
_Avoid_: segmentation loss

**Masked-byte prediction**:
Predicting a masked byte from its surrounding bytes.
→ **Here**: replaces the paper's next-byte cross-entropy, which a bidirectional model
cannot be trained to do without leaking the future into the representation.
_Avoid_: byte LM, next-byte prediction

**Depooling**:
Distributing patch representations back onto the bytes they cover.
_Avoid_: unpooling, scatter

## Measurement

**Fidelity**:
How closely a byteified model reproduces the *source model's* behaviour.
_Avoid_: accuracy, quality, similarity (uncalibrated)

**Retrieval quality**:
How well a model ranks documents for a query, measured against relevance judgements rather
than against any teacher.
_Avoid_: fidelity

**Anisotropy**:
The property of pooled embeddings that they share one dominant direction, so cosine
between them stays near 1.0 regardless of content.
_Avoid_: shared direction (that is the symptom; this is the property)

**Centred cosine**:
Cosine similarity computed after removing the corpus mean, so it responds to the component
that discriminates rather than the one every vector shares.
_Avoid_: raw cosine, plain cosine

**Similarity-matrix correlation**:
Correlation between two models' pairwise cosine matrices — whether they rank documents the
same way, not merely whether their vectors point the same way.
_Avoid_: similarity, agreement