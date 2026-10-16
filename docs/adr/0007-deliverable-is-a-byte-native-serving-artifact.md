# The deliverable is a byte-native serving artifact

byteify stops being a research project that reimplements Bolmo's byteification for encoders
and becomes a tool whose deliverable is a **byte-native serving artifact**: a fixed-shape
encoder graph with a fixed byte axis and a fixed patch budget, deployable without a source
tokenizer. Fidelity to the source model becomes a **constraint the tool preserves and
verifies**, not the benefit it sells.

**Considered Options.** (a) Sell *a better embedder* — the paper's actual claim. Nothing in
this repo demonstrates it: `RESULTS.md` §8 records that no evaluation measures byte-level
competence, Stage 2 is implemented but untrained, and contrastive training is a smoke test.
(b) Sell *a faithful port*. Honest, but fidelity **decreases** as training departs from the
source model, so the ladder would promise progressively worse artifacts as it climbed.
(c) Sell *a byte-native serving artifact* — fully backed by what exists: ONNX export verified
to 3.7e-07 max absolute difference against eager PyTorch, budgeted patching in ADR 0002, and
Stage 1 fidelity measured at 0.89 similarity-matrix correlation.

**Consequences.** Every level's user-facing claim has to be a **measurable deployment
property**, which makes each level gateable and stops the ladder from promising what the
repo cannot yet show. It also excludes Stage 2 from the product ladder: Stage 2 trains the
global model away from the source model, so it violates the fidelity constraint this promise
rests on. Stage 2 stays a research path. The corollary is that the top open problem in
`RESULTS.md` §8.1 — the `L_E` / `L_emb` trade-off, where `λ_emb = 1000` degrades `cos@7` from
0.88 to 0.40 — stays open, because answering it means changing what fidelity is allowed to mean.