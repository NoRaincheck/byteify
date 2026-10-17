# 12. Source models that are not plain checkpoints are resolved, not refused

Date: 2026-10-17

## Status

Accepted.

## Context

`byteify build --model <id>` takes a HuggingFace id and hands it to `AutoModel`. That works
for a plain checkpoint: `config.json` at the repository root, weights beside it, a tokenizer
beside them. Six models were verified this way and `byteify` documents that pointing it at
a different embedding model is a configuration change.

`hotchpotch/bekko-system-one-v0-17m` is not a plain checkpoint. It is a sentence-transformers
repository whose module is custom code:

```
modules.json                              -> the module directory, 0_BekkoInference
0_BekkoInference/backbone_config.json     -> not called config.json
0_BekkoInference/model.safetensors        -> keys prefixed `encoder.backbone.`
0_BekkoInference/heads.{choice,noul,score}-> task heads, beside the encoder
0_BekkoInference/tokenizer/               -> not at the repository root
```

`AutoModel.from_pretrained` cannot read that, and neither can `AutoTokenizer`.

The tempting response is to say this is out of scope: it is a decision model, not an
embedding model, and byteify's verified set is embedding models. That response is wrong for
a specific reason. A byteified encoder delegates its patch representations to the frozen
source encoder -- level 1's fidelity guarantee *is* that delegation -- so the source model is
re-fetched on every `byteify encode` call. A repository byteify can build from but cannot
serve is not a supported target; it would produce an artifact that trains, gates, exports and
then fails on first use. ADR 0007 makes the manifest the only entry point and ADR 0008 makes
ONNX the terminal artifact, and both assume the source model remains reachable from its id.

## Decision

`byteify.hub` resolves both layouts to the same two things -- an `nn.Module` and a tokenizer
-- and names which layout it found.

* Plain checkpoints still go through `AutoModel`. The module layout is a fallback, not a
  replacement, so the ordinary path is unchanged.
* `modules.json` names the module directories; the first carrying a readable config wins.
  A repository with no `modules.json` is treated as already plain.
* Weights are re-homed by stripping a **declared** prefix list
  (`encoder.backbone.`, `encoder.`, `backbone.`, `transformer.`, `model.`), not by a
  heuristic. A key that matches none of them is *reported as dropped* rather than guessed
  at, because the task heads in this repository are genuinely not encoder weights and a load
  that discarded them silently would be indistinguishable from one that found the wrong
  prefix.
* A missing or extra parameter is an error. Loading the overlap would leave part of the
  encoder random, and nothing downstream would notice.
* Both `AutoModel` and `AutoTokenizer` failures are caught. The difference is not a detail:
  a missing file is an `OSError`, while a directory that exists and holds no `config.json`
  is a `ValueError` naming the missing key. Catching only the first meant the module layout
  -- the entire reason this module exists -- never reached the fallback at all.
* Hub ids resolve through `snapshot_download` with an explicit allowlist, so resolution works
  from the cache with `HF_HUB_OFFLINE=1` and does not pull a repository's ONNX exports on
  every serving call.

## Consequences

* `byteify build` and `byteify encode` work for a custom-code module repository, which
  widens what "a configuration change" means.
* The source model still has to be fetchable at serving time. That was already true and is
  now stated in `loader.py` rather than discovered by a reader.
* Weights that no encoder parameter claims are surfaced by name at build time. On this
  model that is the six task-head tensors, and saying so is the difference between a
  documented provenance and a silent one.
* Thresholds remain per source model. Resolving a repository is not measuring it, and
  `bekko-system-one-v0-17m` had to be gated on its own numbers before it could be built --
  see RESULTS.md.