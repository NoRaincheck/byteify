# Stage 2 uses masked-byte prediction, not next-byte prediction

The paper's Stage 2 keeps next-byte cross-entropy. Here Stage 2 keeps the boundary loss and
a **masked**-byte cross-entropy, with the global model unfrozen.

A bidirectional model cannot be trained to predict the next byte. Doing so forces the mask
over future bytes — the embedding is pooled over all positions, so any causal masking leaks
the answer into exactly the representation the model is judged on. Masking is the standard
remedy for bidirectional encoders and keeps the data requirement unchanged: raw text, no
pairs.

This also removes the paper's fused 512-entry vocabulary, where each byte carried a boundary
flag so that boundary prediction was free during sampling. There is no sampling here, so the
prediction target is a masked byte over the 256 real values.

**Considered Options.** Causally masking the byte axis (leaks the future into the pooled
embedding). Keeping next-byte prediction over a causal sub-model (changes what the byteified
model is). Dropping Stage 2's cross-entropy entirely and training contrastively from Stage 1
(which makes the two-stage architecture pointless).