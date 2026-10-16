# All embedding metrics are centred, never raw cosine

Every embedding-level number this project reports is a batch-centred cosine or a
similarity-matrix correlation. Raw cosine between pooled embeddings is never reported as a
result.

Pooled embeddings are strongly anisotropic: they share one dominant direction, so cosine
between any two of them sits near 1.0 almost regardless of content. An early version of
this harness reported a cosine of 0.995 between two models that agreed on nearest
neighbours no better than chance. A metric that reads 0.99 for a degenerate model cannot
support a claim, so it is removed rather than caveated.

**Considered Options.** Reporting raw cosine with a caveat (it reads as a success at a glance
and will be quoted without the caveat). Reporting only similarity-matrix correlation, which
is sound but does not localise *where* two models disagree.

**Consequences.** Numbers here are not comparable to raw-cosine figures in the literature.
Any comparison against an external embedding model must re-centre both sides.