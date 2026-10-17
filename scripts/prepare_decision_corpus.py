"""Materialise a decision model's own workload as byteify reference data and judgements.

Byteification needs raw text, and the text that matters is the text the source model is
asked to embed. For a typed-decision model that is not prose: it is a rendered instruction
and application state, and a rendered candidate description. Training the byte interface on
Project Gutenberg and gating it on decisions would measure a model for a workload it will
never see.

So this renders Open-Jev (``ZefanCai/Open-Jev``) the way ``bekko-system-one`` renders it at
inference time, and writes two files:

* ``--out-texts``  ``{"text": ...}`` per line: rendered queries and candidates, the inputs
  the encoder actually receives. This is what ``byteify build --data`` reads.
* ``--out-judgements`` ``{"query", "candidates", "positive"}`` per line: the same decisions
  with the candidate that was correct, for ``byteify build --judgements``.

The rendering mirrors the model's own layout rather than this repository's convenience:

* query  -- ``Instruction: {instruction}\\nState: {state}``, the ``instruction_state``
  prefix layout, with no system prompt. A Noul decision instead gets the state's JSON with
  the yes/no meanings folded in, exactly as ``input_groups`` composes it, because that is
  what the model is shown for that task.
* candidate -- ``Candidate: {id}: {description}``. Open-Jev stores criterion ids inside
  the option string for Choice (``"billing: Charges, invoices, refunds"``) and not at all
  for Score, so ids are split out where they are present and synthesised where they are
  not. The split is a rendering choice and is stated here because the candidate text ends
  up in the byte stream.

Usage:

    uv run --with pyarrow python scripts/prepare_decision_corpus.py \
        --train-texts 20000 --judgement-decisions 2000

Writes to ``data/corpus/open-jev-<split>.jsonl`` and
``data/judgements/open-jev-<split>.jsonl``. Both are regenerated from the dataset plus a
seed, and both are gitignored: they are data, not code.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DATASET = "ZefanCai/Open-Jev"
CONFIG = "release-v2-redistributable"

#: ``"billing: Charges, invoices, refunds"`` -- the criterion id Open-Jev folds into a
#: Choice option string. Score and Noul options carry no id, so there is nothing to split.
_ID_PREFIX = re.compile(r"^([A-Za-z0-9_.-]+):\s*(\S.*)$", re.DOTALL)


def render_query(question: str, state: str, *, kind: str, options: list[str]) -> str | None:
    """The text the model embeds for a decision, in its own prefix layout.

    ``None`` when a Noul decision's options are not the authored yes/no pair: the model
    folds both meanings into the state beside the state itself, and a Noul whose two
    options cannot be identified has no faithful rendering here.
    """
    if kind == "noul":
        meaning = dict(zip(options, options[:2], strict=False))
        yes = meaning.get("yes") or meaning.get("true")
        no = meaning.get("no") or meaning.get("false")
        if yes is None or no is None:
            return None
        context = json.dumps(
            {"noul": {"yes": yes, "no": no}, "state": json.loads(state)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    else:
        context = state
    return f"Instruction: {question}\nState: {context}"


def render_candidates(kind: str, options: list[str]) -> list[tuple[str, str]]:
    """``(id, description)`` per option, then rendered as ``Candidate: {id}: {text}``."""
    pairs: list[tuple[str, str]] = []
    for index, option in enumerate(options):
        match = _ID_PREFIX.match(option)
        if match:
            pairs.append((match.group(1), match.group(2)))
        elif kind == "noul":
            # Noul ids *are* the authored meanings: "true"/"false" or "yes"/"no".
            pairs.append((option.strip().lower(), option))
        else:
            pairs.append((f"option_{index}", option))
    return pairs


def render_row(row: dict) -> tuple[str, list[str], int | None] | None:
    """One dataset row as ``(query text, candidate texts, gold index)``.

    ``None`` for a row that cannot be rendered, or whose target does not name a single
    correct candidate: a tie has no gold, and inventing one would measure the artifact
    against a label that does not exist.
    """
    options = row.get("options") or []
    target = row.get("target") or []
    if len(options) < 2 or len(target) != len(options):
        return None
    question, state = row.get("question"), row.get("state_json")
    if not question or not state:
        return None
    try:
        json.loads(state)
    except json.JSONDecodeError:
        return None

    candidates_of = render_candidates(row["kind"], options)
    ids = [identifier for identifier, _ in candidates_of]
    if len(set(ids)) != len(ids):
        return None  # duplicate ids would make the candidates indistinguishable

    gold = max(range(len(target)), key=lambda i: target[i])
    if sum(1 for value in target if value == target[gold]) > 1:
        return None

    query = render_query(question, state, kind=row["kind"], options=options)
    if query is None:
        return None
    candidates = [f"Candidate: {identifier}: {text}" for identifier, text in candidates_of]
    return query, candidates, gold


def _load_split(split: str, limit: int | None) -> list[dict]:
    """Read one split's rows straight from the parquet files in the hub cache.

    Deliberately not ``load_dataset``: the streaming loader wants the network, and this
    script has to produce the same bytes on a machine that has already fetched the dataset
    once and a machine that is about to. ``snapshot_download`` resolves against the cache
    first, so the second machine fetches once and is offline afterwards.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            allow_patterns=[f"data/{CONFIG}/{split}-*.parquet"],
        )
    )
    rows: list[dict] = []
    for path in sorted(root.glob(f"data/{CONFIG}/{split}-*.parquet")):
        table = pq.read_table(
            path, columns=["kind", "question", "options", "target", "state_json"]
        )
        rows.extend(table.to_pylist())
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def _write_texts(path: Path, rows: list[dict], wanted: int, rng: random.Random) -> int:
    """Rendered queries and candidates, interleaved, deduplicated, in shuffled order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines: list[str] = []
    for row in rows:
        rendered = render_row(row)
        if rendered is None:
            continue
        query, candidates, _ = rendered
        # Queries are the longer half of what this model embeds and candidates the
        # shorter; taking both, one candidate per query, keeps the mix from becoming a
        # length distribution that happens to suit the byte axis.
        for text in (query, candidates[0]):
            if text not in seen:
                seen.add(text)
                lines.append(json.dumps({"text": text}, ensure_ascii=False))
    rng.shuffle(lines)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines[:wanted]:
            handle.write(line + "\n")
    return len(lines[:wanted])


def _write_judgements(path: Path, rows: list[dict], wanted: int, rng: random.Random) -> dict:
    """Decisions with their candidate set and the correct one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = [render_row(row) for row in rows]
    decisions = [item for item in rendered if item is not None]
    rng.shuffle(decisions)
    kept = decisions[:wanted]
    with path.open("w", encoding="utf-8") as handle:
        for query, candidates, gold in kept:
            handle.write(
                json.dumps(
                    {"query": query, "candidates": candidates, "positive": gold},
                    ensure_ascii=False,
                )
                + "\n"
            )
    n_candidates = sum(len(c) for _, c, _ in kept)
    return {
        "decisions": len(kept),
        "dropped_unrenderable": len(rendered) - len(decisions),
        "mean_candidates": n_candidates / max(len(kept), 1),
        "chance_recall_at_1": sum(1.0 / len(c) for _, c, _ in kept) / max(len(kept), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-texts", type=int, default=20000)
    parser.add_argument("--judgement-decisions", type=int, default=2000)
    parser.add_argument("--limit-rows", type=int, default=40000,
                        help="rows to read per split before rendering")
    parser.add_argument("--out-dir", default="data/corpus")
    parser.add_argument("--judgement-dir", default="data/judgements")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    judgement_dir = Path(args.judgement_dir)

    train = _load_split(args.train_split, args.limit_rows)
    evaluation = _load_split(args.eval_split, args.limit_rows)
    print(f"read {len(train)} train rows, {len(evaluation)} {args.eval_split} rows")

    n_train = _write_texts(out_dir / f"open-jev-{args.train_split}.jsonl", train,
                           args.train_texts, rng)
    print(f"wrote {n_train} reference texts -> {out_dir}/open-jev-{args.train_split}.jsonl")

    summary = _write_judgements(
        judgement_dir / f"open-jev-{args.eval_split}.jsonl",
        evaluation,
        args.judgement_decisions,
        rng,
    )
    print(
        f"wrote {summary['decisions']} judgement decisions -> "
        f"{judgement_dir}/open-jev-{args.eval_split}.jsonl "
        f"(chance recall@1 {summary['chance_recall_at_1']:.3f}, "
        f"{summary['dropped_unrenderable']} rows dropped as unrenderable or tied)"
    )


if __name__ == "__main__":
    main()