# Budgeted patching, with a fixed patch budget

At inference a byteified encoder emits a fixed number of patches — the most confident
boundary positions — and masks out any that fell below threshold.

ONNX graphs cannot have data-dependent output shapes, and the patch count is exactly that:
training pools wherever the boundary predictor fires. Selecting the top-scoring positions
uses operators ONNX supports, and gives a defensible rule rather than a workaround —
a model confident about fewer boundaries than the budget leaves slots masked out, and one
confident about more drops the least confident and merges their patches.

**Considered Options.** Exporting a dynamic-length graph (not representable in ONNX). Padding
to a maximum observed length, which leaves the graph correct but lets the boundary predictor
itself decide the shape — the same problem one level up. Fixing the boundary predictor's
output length instead, which is what this is.

**Consequences.** The exported graph's byte axis is fixed, so inputs must be padded to a
single length. Above the source tokenizer's own compression ratio, patches and tokens no
longer correspond one-to-one, which makes per-patch comparison against the token axis
undefined and is reported as such.