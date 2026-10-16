# ONNX is the terminal artifact

byteify's deliverable is an ONNX graph plus its external data, wrapped in a self-describing
manifest, and nothing else. No HuggingFace-loadable directory, no `save_pretrained` pair.

ONNX is already the only format in this repo verified against eager PyTorch (3.7e-07 max
absolute difference, `export_onnx.py`), and it is the format that *forced* budgeted patching
(ADR 0002) — a fixed patch count is what makes the guarantee in ADR 0007 expressible at all.
A fixed byte axis and a fixed patch budget cannot be carried by a `transformers`-style
directory, because those formats have no way to say "this axis is static and this is the
budget."

**Considered Options.** (a) A HuggingFace-loadable directory, so the model loads through
`transformers` or `sentence-transformers` — real value for anyone already working that way,
but it means owning a compatibility class for architectures outside the verified six-model
suite, and it still could not carry a static patch budget. (b) Both, with ONNX guaranteed —
pays the maintenance cost of (a) anyway for a second format the promise in ADR 0007 is not
stated in terms of.

**Consequences.** The manifest becomes load-bearing, because the local-encoder architecture is
currently written to `args.json` and **never read back** — `export_onnx.py:64` and
`train_contrastive.py:103` both hardcode `LocalContextEncoder(n_layers=1, ffn_hidden=1024,
window=128)` regardless of what the checkpoint was trained with. Anyone loading a byteify
artifact without the manifest cannot reconstruct the model at all. The manifest must carry the
local-encoder architecture, patch budget, max bytes, source model id and revision, and the
gate results for the level that produced it.