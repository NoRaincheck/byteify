# An embedding alignment loss replaces likelihood distillation

Stage 1 adds a loss aligning the byteified model's pooled sentence embedding with the
source model's, by batch-centred cosine. The weights are 4 for the boundary loss, 1 for the
stitching loss, 1 for this one.

The paper's Stage 1 is provably minimised exactly when the byte model emulates the subword
model, because it has an output distribution to match against. An embedder has none, so
"recover the source model's behaviour" becomes a choice rather than a derivation. Measured
on held-out data, the stitching loss alone plateaus at a similarity-matrix correlation of
about 0.34; adding this loss takes it to about 0.89. The stitching loss is the wrong
objective for an embedding model because Bolmo is rescued by a complementary output-side
term that has no analogue here.

The two losses genuinely compete — weighting this one heavily improves centred cosine while
degrading per-patch fidelity — so the mixture is a tuning question, not a settled one.

**Considered Options.** Stitching loss alone, which imitates per-patch behaviour but not the
embedding geometry. Training the whole contrastive stack inside Stage 1, which optimises the
end goal but discards the behavioural fidelity that makes Stage 1 a useful architecture
screen. A whitened stitching loss, which might close the gap without a second term; untested.