"""Grade a byteified artifact inside the decision path it is meant to stand in for.

The fidelity gates measure agreement between two *embedding spaces*. For
`bekko-system-one-v0-17m` that is the wrong instrument, and this repository's own numbers
say so: the source model's mean-pooled-embedding retrieval on Open-Jev is 0.262 against a
0.419 chance level, because ranking candidates by cosine between pooled embeddings is not
how this model decides. It decides through a cross-attention encoder and three task heads.

So this harness replaces the embedding-space proxy with the model's own decision function.
Read `inference_v0.py`, and the path is:

    prefix_ids  = tokenised query (instruction + state), CLS/SEP bracketed
    doc_ids     = tokenised candidates ("Candidate: {id}: {description}")
    hidden      = encoder(backbone)(prefix_ids, prefix_mask, doc_ids, doc_mask, owners)
                  -- the prefix is encoded alone, and every candidate's tokens attend to
                     [its query's keys, its own keys], so a candidate is scored *in the
                     context of* its query
    pooled      = masked mean over the candidate's token positions
    scores      = heads[task](pooled)          -- one scalar per candidate
    probs       = softmax(scores) within the decision
    choice -> argmax, noul -> probability of the authored "yes", score -> Σ p·value

Two paths run through that same function:

* **source** -- the subword token stream the model was trained on. Its output is the
  reference, and it is produced by the model's own `predict_groups`, not by a
  reimplementation here.
* **byteified** -- the artifact's pooled embedding for the query and for each candidate,
  each standing as a one-token sequence. The backbone's layers, the cross-attention, the
  pooling and the heads are all the source model's own weights, untouched; only the
  representation the bytes produce is substituted. Positions are laid out as the model
  lays them out (prefix at 0, candidates at 1..k).

The byteified path is therefore a *fidelity probe of the decision path*, not a claim that
the artifact is a drop-in decision model: a candidate is scored from one pooled vector
rather than from its tokens attending to the query's tokens. It is the weakest thing the
artifact can be asked to do, which makes it the right thing to ask.

The harness refuses to report byteified numbers unless its own source path reproduces the
model's `predict_groups` to `--tolerance`. A harness that disagreed with the model it is
measuring would be the same defect the gates exist to catch, one level up.

    uv run --with sentence-transformers --with pyarrow python scripts/decision_harness.py \
        --artifact artifacts/bekko-17m-faithful-2k \
        --judgements data/judgements/open-jev-validation.jsonl --limit 400

`--judgements` supplies decisions whose queries and candidates are already rendered; the
structured Open-Jev row is recovered for the source path so that both paths see the same
decision. Requires `sentence-transformers` because the model's own module is what runs the
decision path -- that is the point of the harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ID = "hotchpotch/bekko-system-one-v0-17m"
OPEN_JEV = "ZefanCai/Open-Jev"


def load_reference_module():
    """The model's own inference module, from the hub cache, without a network."""
    import importlib

    import huggingface_hub

    root = huggingface_hub.snapshot_download(REPO_ID, allow_patterns=["inference_v0.py"])
    sys.path.insert(0, root)
    return importlib.import_module("inference_v0")


def open_jev_rows(split: str):
    """Rows of one Open-Jev split, from the parquet in the local cache."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        OPEN_JEV, repo_type="dataset", allow_patterns=[f"data/release-v2-redistributable/{split}-*.parquet"]
    )
    files = sorted(Path(root).glob(f"data/release-v2-redistributable/{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {split} parquet in the Open-Jev cache; run prepare_decision_corpus.py")
    return pq.read_table(
        files[0],
        columns=["id", "kind", "question", "options", "target", "state_json", "metadata_json"],
    ).to_pylist()


def native_input(row, render_candidates):
    """One Open-Jev row as the model's native request object.

    Criterion ids come from the same derivation the reference corpus uses, so the candidate
    text the byteified path embeds is byte-for-byte the text the source path tokenises.
    Score criteria carry the numeric rubric value, which the model requires and which
    Open-Jev keeps in metadata.
    """
    pairs = render_candidates(row["kind"], row["options"])
    values = json.loads(row["metadata_json"])["score_values"] if row["kind"] == "score" else None
    criteria = [
        {"id": identifier, "description_json": json.dumps(text), "value": value}
        for (identifier, text), value in zip(pairs, values or [None] * len(pairs), strict=False)
    ]
    return {
        "state_json": row["state_json"],
        "decisions": [
            {
                "id": "decision",
                "kind": "judgment",
                "type": row["kind"],
                "instructions_json": json.dumps(row["question"]),
                "system_prompt": "",
                "criteria": criteria,
                "documents": [],
                "scoring": None,
            }
        ],
    }


class ByteEmbeddings(torch.nn.Module):
    """A backbone facade whose embedding lookup returns supplied vectors.

    `_SDPAPrefix.forward` reaches into four things on its backbone: `embeddings`, `layers`,
    `rotary_emb`, `final_norm`. Replacing `embeddings` with this is what lets the model's
    own cross-attention run over byteified pooled vectors without a line of its attention
    maths being copied here -- and a copy would be a second implementation to keep in step
    with the first, which is how a harness starts lying.

    It has to be an `nn.Module` rather than a plain object: the encoder assigns this onto
    its own `backbone` attribute, and `nn.Module.__setattr__` refuses anything that is not a
    module in that slot.
    """

    def __init__(self, backbone):
        super().__init__()
        # Not registered as a child: the real backbone is already owned by the encoder, and
        # registering it twice would put its parameters in two state dicts.
        object.__setattr__(self, "_inner", backbone)
        #: Keyed on the identity of the ids tensor the forward passes, because
        #: `_SDPAPrefix.forward` calls `embeddings(prefix_ids)` and then
        #: `embeddings(doc_ids)`, and the two must not be confused for one another.
        self.vectors = {}

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)

    def embeddings(self, ids=None):
        try:
            return self.vectors[id(ids)]
        except KeyError:
            raise RuntimeError(
                "the decision path asked for embeddings of an id tensor this harness did "
                "not supply; supplying them by call order would be a guess"
            ) from None


@torch.no_grad()
def byteified_scores(runtime, embed_texts, query, candidates, task):
    """The source model's own decision path, fed byteified pooled vectors.

    `embed_texts(texts) -> [n, d]`: the artifact's own embedding, already final-norm pooled
    and normalised. Each vector enters as a one-token sequence, so the masked mean over a
    candidate's positions is the identity and the head reads the vector directly.
    """
    backbone = runtime.encoder.backbone
    vectors = embed_texts([query] + list(candidates))
    k = len(candidates)
    # The decision path's batch axis is one row per candidate, not per decision: every
    # candidate's tokens attend to `[its query's keys, its own keys]`, so the model lays out
    # one row per candidate and repeats its query through `owners`. Here every row is a
    # single token: the query at position 0, and candidate i at position i + 1.
    prefix = vectors[:1].expand(k, 1, -1).contiguous()
    docs = vectors[1:].view(k, 1, -1)
    ones_prefix = torch.ones(k, 1, dtype=torch.bool)
    ones_docs = torch.ones(k, 1, dtype=torch.bool)

    encoder = runtime.encoder
    original = encoder.backbone
    facade = ByteEmbeddings(backbone)
    facade.vectors = {id(prefix): prefix, id(docs): docs}
    try:
        encoder.backbone = facade
        hidden = encoder(
            prefix_ids=prefix,
            prefix_mask=ones_prefix,
            doc_ids=docs,
            doc_mask=ones_docs,
            owners=torch.zeros(k, dtype=torch.long),
        )
    finally:
        encoder.backbone = original

    pooled = (hidden * ones_docs.unsqueeze(-1)).sum(1) / ones_docs.sum(1)[:, None].float()
    return runtime.heads[task](pooled.float()).flatten()


def truncated_group(group, budget_bytes, mod):
    """The same decision, with the query cut to what the artifact's byte axis can hold.

    The control that makes the comparison readable. Without it, "the byteified artifact
    scores lower" has two indistinguishable causes -- that byteification lost something, and
    that a fixed byte axis discarded half of every query -- and only one of them is a
    property of byteification. Truncation is applied the way the model applies its own: a
    budget shared between instruction and context, unbalanced only when one runs out.

    Candidates are truncated the same way. Both are rebuilt through the model's own
    `QueryParts`, so the tokenisation, the brackets and the layout are unchanged.
    """
    import unicodedata

    def cut(text, limit):
        raw = unicodedata.normalize("NFC", text).encode("utf-8")[: max(0, limit)]
        return raw.decode("utf-8", errors="ignore")

    parts = group.query_parts
    instruction_bytes = len(parts.instruction.encode("utf-8"))
    context_bytes = len(parts.context.encode("utf-8"))
    # The model's own policy is half per field, then transfer what the other did not use.
    share = max(budget_bytes // 2, 0)
    given_instruction = min(instruction_bytes, share)
    given_context = min(context_bytes, budget_bytes - given_instruction)
    spare = budget_bytes - given_instruction - given_context
    given_instruction += min(instruction_bytes - given_instruction, spare)

    tightened = mod.QueryParts(
        instruction=cut(parts.instruction, given_instruction) or parts.instruction[:1],
        context=cut(parts.context, given_context),
        system=parts.system,
        layout=parts.layout,
    )
    return mod.Group(
        tightened.render(),
        [cut(candidate, budget_bytes) for candidate in group.candidates],
        group.task,
        target=group.target,
        query_parts=tightened,
        metadata=group.metadata,
    )


def _softmax(scores: torch.Tensor) -> torch.Tensor:
    return torch.softmax(scores.double(), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--judgements", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--tolerance", type=float, default=2e-3,
                        help="max |probability| disagreement with the model's own "
                             "predict_groups before the harness refuses to report")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_decision_corpus import render_candidates

    from byteify.inference import ByteifiedEmbedder
    from byteify.manifest import Manifest
    from byteify.training import load_judgements

    mod = load_reference_module()
    judgements = load_judgements(args.judgements)

    rows = {}
    for row in open_jev_rows(args.split):
        rows[row["id"]] = row
    # The judgements file does not carry the dataset row id, so match on the rendered
    # query the two renderers agree on. Anything unmatched is a bug in one of them and is
    # reported rather than skipped.
    by_query = {}
    for row in rows.values():
        from prepare_decision_corpus import render_row

        rendered = render_row(row)
        if rendered is not None:
            by_query.setdefault(rendered[0], row)

    selected, missing = [], 0
    for index, (query, candidates, relevant) in enumerate(
        zip(judgements.queries, judgements.candidates, judgements.relevant, strict=True)
    ):
        if index >= args.limit:
            break
        row = by_query.get(query)
        if row is None:
            missing += 1
            continue
        # `relevant` holds index lists; a decision graded against a list rather than the
        # index inside it compares unequal to every candidate, which reads as a model that
        # is wrong on every decision rather than as a bug in the harness.
        selected.append((row, query, candidates, relevant[0]))
    if missing:
        print(f"warning: {missing} judgement queries had no matching dataset row and were dropped")
    if not selected:
        raise SystemExit("no judgement query matched a dataset row; the two renderers disagree")

    print(f"model       {REPO_ID}")
    print(f"decisions   {len(selected)} from the {args.split} split")

    # -- the reference, from the model's own code ------------------------------
    model = mod.BekkoSentenceTransformer(REPO_ID, trust_remote_code=True, device=args.device)
    runtime = model[0]
    groups = [mod.input_groups(native_input(row, render_candidates))[0] for row, *_ in selected]
    reference = model.predict_groups(groups, batch_size=32, show_progress_bar=False)
    print(f"reference   the model's own predict_groups, {len(reference)} probability vectors")

    # -- the artifact, loaded through its manifest -----------------------------
    manifest = Manifest.load(args.artifact)
    embedder = ByteifiedEmbedder.load(args.artifact, device=args.device).eval()
    axis = manifest.max_bytes
    print(f"artifact    {args.artifact}")
    print(f"  source    {manifest.source_id}")
    print(f"  axis      {axis} bytes, patch budget {manifest.budget}")
    print(f"  trie      {manifest.trie.entries} entries, coverage "
          f"{manifest.trie.coverage if manifest.trie.coverage is not None else float('nan'):.3f}")

    # -- run both paths -------------------------------------------------------
    rule = manifest.stream_rule()

    def embed_texts(texts):
        """The artifact's embedding, truncating to its axis and recording what was lost."""
        clipped = [text if _fits(rule, text, axis) else _truncate(text, axis) for text in texts]
        # Accumulated across every call, not read off the last one: an artifact whose axis
        # truncates the first hundred queries and fits the last one would otherwise report
        # that it dropped nothing.
        embed_texts.dropped = max(
            getattr(embed_texts, "dropped", 0),
            *(
                _content_bytes(rule, text) - _content_bytes(rule, cut)
                for text, cut in zip(texts, clipped, strict=True)
            ),
        )
        return embedder.encode_text(clipped)

    worst_reference_gap = 0.0
    rows_report = []
    for (row, query, candidates, gold), group, reference_p in zip(
        selected, groups, reference, strict=True
    ):
        source_scores = byteified_scores_source(runtime, group)
        source_p = _softmax(source_scores)
        gap = float((source_p - reference_p.double()).abs().max())
        worst_reference_gap = max(worst_reference_gap, gap)

        # The truncation control: the source model, on the bytes the artifact can see.
        tight = truncated_group(group, axis - 2, mod)
        clipped_p = _softmax(byteified_scores_source(runtime, tight))

        byte_p = _softmax(byteified_scores(runtime, embed_texts, query, candidates, row["kind"]))

        rows_report.append({
            "kind": row["kind"],
            "n_candidates": len(candidates),
            "gold": gold,
            "reference_pick": int(reference_p.argmax()),
            "source_pick": int(source_p.argmax()),
            "clipped_pick": int(clipped_p.argmax()),
            "byte_pick": int(byte_p.argmax()),
            "reference_p": reference_p.tolist(),
            "byte_p": byte_p.tolist(),
        })

    if worst_reference_gap > args.tolerance:
        raise SystemExit(
            f"the harness's source path disagrees with the model's own predict_groups by "
            f"{worst_reference_gap:.2e} (tolerance {args.tolerance:.1e}); refusing to report "
            "byteified numbers from a harness that does not reproduce the model it measures"
        )
    print(f"  check     own source path reproduces predict_groups to {worst_reference_gap:.2e} "
          f"(tolerance {args.tolerance:.1e})")
    print(f"  byte      up to {getattr(embed_texts, 'dropped', 0)} bytes dropped per text by "
          f"the {axis}-byte axis")

    # -- report ---------------------------------------------------------------
    print()
    print(f"{'kind':7} {'n':>4} {'chance':>7} {'source':>8} {'src@axis':>9} {'byteified':>10} "
          f"{'agree':>7} {'meanTV':>7}")
    print("-" * 72)
    totals = {}
    for kind in sorted({row["kind"] for row in rows_report}):
        group_rows = [row for row in rows_report if row["kind"] == kind]
        # Chance is 1/k averaged over the decisions in this group, computed the same way
        # for every group rather than derived from the model's picks: a chance level that
        # depends on the model being graded cannot be read as a baseline.
        chance = sum(1.0 / row["n_candidates"] for row in group_rows) / len(group_rows)

        def accuracy(key, rows=group_rows):
            return sum(row[key] == row["gold"] for row in rows) / len(rows)
        agree = sum(row["source_pick"] == row["byte_pick"] for row in group_rows) / len(group_rows)
        tv = sum(_total_variation(row["reference_p"], row["byte_p"]) for row in group_rows) / len(group_rows)
        print(f"{kind:7} {len(group_rows):>4} {chance:>7.3f} {accuracy('source_pick'):>8.3f} "
              f"{accuracy('clipped_pick'):>9.3f} {accuracy('byte_pick'):>10.3f} "
              f"{agree:>7.3f} {tv:>7.3f}")
        totals[kind] = {
            "n": len(group_rows), "chance": chance,
            "source": accuracy("source_pick"), "source_at_axis": accuracy("clipped_pick"),
            "byteified": accuracy("byte_pick"), "agreement": agree, "mean_tv": tv,
        }

    def overall(key):
        return sum(row[key] == row["gold"] for row in rows_report) / len(rows_report)

    overall_chance = sum(1.0 / row["n_candidates"] for row in rows_report) / len(rows_report)
    overall_agree = sum(row["source_pick"] == row["byte_pick"] for row in rows_report) / len(rows_report)
    overall_tv = sum(_total_variation(row["reference_p"], row["byte_p"]) for row in rows_report) / len(rows_report)
    print("-" * 72)
    print(f"{'all':7} {len(rows_report):>4} {overall_chance:>7.3f} {overall('source_pick'):>8.3f} "
          f"{overall('clipped_pick'):>9.3f} {overall('byte_pick'):>10.3f} "
          f"{overall_agree:>7.3f} {overall_tv:>7.3f}")

    print()
    print("chance     1/k averaged over each decision's own candidate pool")
    print("source     the model's own predict_groups on the full text")
    print("src@axis   the same model on the bytes the artifact's byte axis can hold -- the")
    print("           truncation control, separating the axis from byteification")
    print("byteified  the artifact's pooled vectors through the same backbone and heads")
    print("agree      the byteified path picking the same candidate as the source path")
    print("meanTV     mean total-variation distance between the two probability vectors")

    report = {
        "artifact": str(args.artifact),
        "source_id": manifest.source_id,
        "axis": axis,
        "budget": manifest.budget,
        "decisions": len(rows_report),
        "reference_gap": worst_reference_gap,
        "per_kind": totals,
        "all": {"chance": overall_chance, "source": overall("source_pick"),
                "source_at_axis": overall("clipped_pick"),
                "byteified": overall("byte_pick"), "agreement": overall_agree,
                "mean_tv": overall_tv},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}")


def byteified_scores_source(runtime, group):
    """The source model's own forward for one group, from its collated features.

    The same two calls `predict_groups` makes internally, so the probabilities this harness
    grades against are the ones the model produces -- which is what the reference check
    below is asserting, and what makes a disagreement here a real finding rather than a
    difference in plumbing.
    """
    from inference_v0 import collate_groups, to_device

    prepared = runtime.prepare_groups([group])
    device = next(runtime.parameters()).device
    return runtime(to_device(collate_groups(prepared, runtime), device))["scores"].flatten()


def _total_variation(p, q) -> float:
    a, b = torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
    return float((a - b).abs().sum() / 2)


def _content_bytes(rule, text: str) -> int:
    return len(rule.build(text, 10_000)) - len(rule.pseudo_bytes)


def _fits(rule, text: str, axis: int) -> bool:
    return _content_bytes(rule, text) <= axis - len(rule.pseudo_bytes)


def _truncate(text: str, axis: int) -> str:
    """Cut text to what the axis holds, on a byte boundary that does not split a character."""
    import unicodedata

    budget = max(0, axis - 2)
    raw = unicodedata.normalize("NFC", text).encode("utf-8")[:budget]
    return raw.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    main()