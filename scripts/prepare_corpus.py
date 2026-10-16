"""Materialise training corpora as local shards.

Byteification needs raw text, not tokens: the whole point is that the model never
sees a subword tokenizer. This streams a dataset from the Hub and writes plain text
shards to disk so training runs are reproducible and resumable.

    uv run --with datasets python scripts/prepare_corpus.py --dataset fineweb-edu --mb 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SOURCES = {
    # name -> (hub id, config, split, text column)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"),
    "code": ("bigcode/the-stack-smol", "default", "train", "content"),
    "japanese": ("allenai/c4", "ja", "train", "text"),
    "chinese": ("allenai/c4", "zh", "train", "text"),
    "russian": ("allenai/c4", "ru", "train", "text"),
    "arabic": ("allenai/c4", "ar", "train", "text"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fineweb-edu", choices=sorted(SOURCES))
    parser.add_argument("--mb", type=float, default=100.0, help="target shard size in MB")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    hub_id, config, split, column = SOURCES[args.dataset]
    out_dir = Path("data/corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{args.dataset}.jsonl"
    if out_path.exists():
        print(f"{out_path} already exists ({out_path.stat().st_size / 1e6:.1f} MB); delete it first")
        return

    target_bytes = int(args.mb * 1e6)
    print(f"streaming {hub_id} ({config}/{split}) -> {out_path} ({args.mb} MB)")

    ds = load_dataset(hub_id, config, split=split, streaming=True)
    t0 = time.time()
    written = 0
    n_docs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(column)
            if not text:
                continue
            text = text.strip()
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({"text": text[: args.max_chars]}, ensure_ascii=False) + "\n")
            written += len(text)
            n_docs += 1
            if written >= target_bytes:
                break
            if n_docs % 20000 == 0 and n_docs:
                rate = written / max(time.time() - t0, 1e-6) / 1e6
                print(
                    f"  {n_docs:>8,} docs  {written / 1e6:8.1f} MB"
                    f"  ({rate:.1f} MB/s)",
                    flush=True,
                )

    size_mb = out_path.stat().st_size / 1e6
    print(f"done: {n_docs:,} docs, {size_mb:.1f} MB in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()