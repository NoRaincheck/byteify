# One byte-stream rule, shared by training and inference

The byte stream is produced by **one** constructor, used by every path. Conversion — the
offline step that turns a source model into an artifact — requires the source tokenizer, because
the suffix trie is built from its vocabulary and there is no way around that. **Inference does
not.** Inference consumes the prebuilt trie from the artifact manifest and applies a declared
normalisation rule recorded in that same manifest.

Why: the question "does NFC normalisation actually matter?" is only load-bearing if training and
inference can disagree about the byte stream. Make them agree by construction and the question
stops blocking anything — it becomes the value of a declared setting. Today the paths disagree
in two ways. `inference.py:129` derives its bytes from `align_text`, which calls the tokenizer,
even though the exported graph's three inputs do not; and training runs
`add_special_tokens=True` while inference runs `False`, so pseudo-bytes for `[CLS]`/`[SEP]`
appear in training and never at inference, against a `n_byte_values = 256 + len(specials)`
byte table reserved for them.

**Considered Options.** (a) Keep deriving bytes from the tokenizer at inference. Simplest, and it
preserves whatever the tokenizer does — but it leaves the tokenizer in the serving path, which
is the thing ADR 0007 promises to remove. (b) Caller supplies `byte_ids`. Honest, and it makes
byteify a graph rather than an embedder. (c) One declared rule, applied by one constructor, with
the tokenizer required only at conversion time.

**Consequences.** Normalisation becomes a **declared, per-tokenizer-family setting with a
default** — reproducible and overridable, rather than an emergent property of whichever code
path ran. The rule travels in the manifest alongside the trie, the patch budget and the
architecture, so an artifact carries everything needed to reproduce its own inputs; this is
another reason the manifest from ADR 0008 is load-bearing. Train/inference consistency stops
being a property that two code paths have to be kept in agreement about by hand, which is how
the special-token asymmetry arose in the first place.

**Measured defaults.** The rule is set per family from measurement, not from argument.

- **Byte-level BPE** — `unicodedata.normalize("NFC", text).encode("utf-8")`. Exact: 158,867
  codepoints swept, every adversarial sample matched, NFKC explicitly ruled out. Minimum
  dependency on the tokenizer: **none**. `byte_ids_for_inference` (`inference.py:129`) is
  making a tokenizer call that is provably redundant.
- **WordPiece** — `text.encode("utf-8")`. byteify already produces this, so no normalisation is
  needed for the byte stream itself.
- **SentencePiece** — raw bytes, **conditional on the alignment fix below**.

**The alignment is not lossless today.** `alignment.py:170-188` clamps `start_byte` backwards
with `min(char_start_b, cursor)` to guarantee every token covers at least one byte, but
HuggingFace's SentencePiece offsets *overlap* — so bytes get re-emitted. `xlm-roberta-base` on
`Ω` produces `b'\xce\xa9\xce\xa9'`; `日本語のテキスト` becomes `日日本…`. **63% of corpus samples are
affected, and the stream is always longer than the raw text.** The same mechanism hits WordPiece
Hangul (24 bytes for 9) and 2.56% of byte-level BPE samples. `RESULTS.md:313` claims losslessness
only for byte-level BPE — the one family whose offsets are disjoint — and `xlm-roberta-base` was
never trained, which is why this has stayed invisible.

**The special-token asymmetry is now characterised.** Training emits exactly two pseudo-bytes,
`[CLS]` and `[SEP]`, always at positions `[0, N-1]`, with `boundary=1`. Inference emits none. The
remaining 3–5 reserved byte values (`[PAD]`, `[MASK]`, `[UNK]`) are **never emitted by any path**,
and `losses.sample_mlm_mask`'s `special_prob` parameter substitutes a random byte rather than a
special, so it is dead. The one-constructor rule must therefore decide once whether pseudo-bytes
bracket the stream; to preserve the existing Stage 1 checkpoint it has to be *yes at both ends*.