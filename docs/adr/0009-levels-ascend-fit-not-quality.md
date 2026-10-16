# Byteification levels ascend fit to the workload, not quality

Every level keeps a Stage 1 checkpoint, so fidelity to the source model is constant across the
whole ladder. What changes is how well the artifact fits the workload it was fitted for: the
base level is a faithful byte-native port, the next adds contrastive training on the user's
own corpus, the top merges a post-trained source delta (task arithmetic) into the body.

**Considered Options.** (a) **Fit.** Every rung preserves fidelity, every rung is measurable,
and the reference data the user supplies determines which rung is reachable — which is why
"reference data" is part of the contract at all rather than an optional extra. (b) **Capability.**
Ascend toward byte-level competence. Ruled out by ADR 0007: Stage 2 trains the global model away
from the source model, so it violates the fidelity constraint the whole promise rests on.
(c) **Efficiency.** Ascend toward smaller or faster artifacts. Rejected as a ladder because
patch budget and byte-axis length are knobs available at *every* rung, not rungs themselves.

**Consequences.** Stage 2 is not a level and is not on the product path — which is also why
ADR 0007 leaves the `L_E` / `L_emb` trade-off open, since answering it means changing what
fidelity is allowed to mean. Task arithmetic re-enters at the top rung not because it improves
the model but because it is the one operation that requires Stage 1 to have left the global
model bit-identical to the source (`task_arithmetic.py:9-12`). Each rung must ship a gate that
measures fit, and the middle rung's gate needs relevance data that the bottom rung does not.