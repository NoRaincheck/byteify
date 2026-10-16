# Standalone package, not a fork of bolmo-core

This is a separate package that reimplements byteification for encoders, rather than a fork
of `allenai/bolmo-core` with the encoder path added.

The upstream framework cannot host the port without changes at every level: it has no
ModernBERT, its model requires an LM head, its local decoder exists only for next-byte
prediction, and it has no contrastive or embedding train module. Adapting it would have
meant porting a transformer family into it, adding a parallel model class, and adding a
train module — all inside a fork that would then need rebasing.

The upstream code was still read closely and the algorithm ported faithfully, including
details the paper omits: the suffix trie is built in reverse byte order so a backwards
suffix walk descends it, and the boundary predictor sits on the local encoder's contextual
output rather than being causal itself.

**Considered Options.** Forking bolmo-core and adding a ModernBERT block plus an encoder
train module (permanent rebasing cost, and the decoder-only assumptions run through the
framework). Depending on `olmo-core` and implementing only the byte-level parts on top
(keeps a dependency whose core assumptions still do not hold). Reimplementing from the paper
alone (would have missed the two implementation details above).