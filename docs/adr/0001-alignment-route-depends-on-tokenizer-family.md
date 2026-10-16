# Alignment route depends on tokenizer family

The byte stream is derived differently depending on which tokenizer family the source
model uses: byte-level BPE by concatenating per-token byte strings, everything else from
the fast tokenizer's offset mapping.

Offsets are untrustworthy for byte-level BPE with multi-byte characters — for
`日本語のテキスト` the tokenizer reports overlapping duplicate spans `(2,3), (2,3)` for three
consecutive tokens, and following them duplicates characters. Offsets are correct for
WordPiece and SentencePiece, whose pre-tokenizers drop whitespace and so need gap
absorption instead.

A single route would be simpler but would silently corrupt one family or the other. The
family is detected once per tokenizer and the choice is validated by tests that assert the
two routes agree where both are valid.

**Considered Options.** Trusting offsets everywhere (corrupts byte-level BPE on CJK).
Concatenating token bytes everywhere (loses bytes where a pre-tokenizer drops whitespace).
Rejecting tokenizers without exact alignment, which would have excluded every uncased
model.