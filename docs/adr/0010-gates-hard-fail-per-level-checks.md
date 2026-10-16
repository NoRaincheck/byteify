# Gates hard-fail, and each level declares its own checks

A byteification level's gate is not a warning. Failing it stops the command and **no artifact is
written**. What the gate checks is declared by the level itself: numerical agreement between
the ONNX graph and eager PyTorch at every level, fidelity to the source model at the levels
that preserve it, and retrieval at the levels whose guarantee is fit.

Why hard fail rather than warn: the repo's only automated check today is `export_onnx.py:158`,
and a model with a **randomly initialised** byte interface passes it. Boundary accuracy 0.000,
centred cosine −0.008, similarity-matrix correlation 0.145 — and it exports cleanly and matches
eager PyTorch. A gate that warns reproduces precisely the failure the gate exists to prevent.

**Considered Options.** (a) Warn and emit anyway — cheap, never blocks a user, defeats the
purpose. (b) Hard fail on one global check — simple, but then a single set of thresholds has to
serve rungs whose guarantees differ in kind: fidelity at the base level, retrieval at the fitted
one. (c) Hard fail, with each level declaring its own checks.

**Consequences.** Every threshold becomes a published contract — someone will run a level on a
corpus where retrieval cannot beat chance and be told no. That is the intended behaviour, but it
means thresholds must be derived from measured distributions rather than picked. It also means
the base level's fidelity thresholds are **per source model**: `sim_corr` on `ettin-encoder-17m`
is not automatically the right number for `xlm-roberta-base`, so shipping the base level means
publishing a per-model threshold table rather than one number.