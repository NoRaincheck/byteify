"""``byteify`` -- the command. One documented surface, a level, and an artifact.

    byteify build   --model ID --level faithful --data PATH... --out DIR
    byteify encode  ARTIFACT_DIR --text "..."        # or --file docs.txt
    byteify inspect ARTIFACT_DIR
    byteify levels

Everything else -- resume, merge, internals -- sits behind these and is not part of the
documented surface.

Two properties are load-bearing:

* **Gates hard-fail and write nothing.** If a check fails the command stops and the output
  directory is left empty. An artifact that failed its own promise is worse than none: it
  loads, it runs, it is wrong.
* **The manifest is the only entry point.** ``encode`` takes a directory and nothing else,
  and runs with no source tokenizer present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _say(message: str = "") -> None:
    print(message, flush=True)


def _plan(steps: list[tuple[str, float]]) -> None:
    """Say what is about to happen and roughly how long, so a run is not a black box."""
    _say()
    _say("plan:")
    for label, seconds in steps:
        minutes = seconds / 60
        when = f"~{seconds:.0f}s" if seconds < 90 else f"~{minutes:.1f} min"
        _say(f"  {label:<46} {when:>9}")
    _say()


# -- build -----------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from .build import build_artifact

    try:
        manifest = build_artifact(
            model=args.model,
            level=args.level,
            data=args.data,
            output_dir=Path(args.out),
            steps=args.steps,
            batch_size=args.batch_size,
            max_bytes=args.max_bytes,
            patch_budget=args.patch_budget,
            local_layers=args.local_layers,
            local_ffn=args.local_ffn,
            local_window=args.local_window,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            judgements=args.judgements,
            delta_from=args.delta_from,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
    except (KeyError, FileNotFoundError, ValueError, RuntimeError) as exc:
        _say(f"error: {exc}")
        return 1

    if manifest is None:
        return 0  # --dry-run

    _say()
    _say(f"artifact: {Path(args.out).resolve()}")
    _say(f"  level       {args.level}")
    _say(f"  source      {manifest.source_id}")
    _say(f"  byte axis   {manifest.max_bytes}, patch budget {manifest.budget}")
    _say(f"  byte stream {manifest.family} / {manifest.byte_stream_rule}")
    if manifest.trie.coverage is not None:
        _say(f"  trie        {manifest.trie.entries} entries, "
             f"{manifest.trie.coverage * 100:.1f}% coverage")
    return 0


# -- encode ----------------------------------------------------------------


def cmd_encode(args: argparse.Namespace) -> int:
    import torch

    from .inference import ByteifiedEmbedder
    from .manifest import Manifest

    artifact = Path(args.artifact)
    try:
        manifest = Manifest.load(artifact)
    except Exception as exc:  # noqa: BLE001 - surfaced as a message, not a traceback
        _say(f"error: {exc}")
        return 1

    if args.text is not None:
        texts = [args.text]
    elif args.file:
        texts = [t for t in Path(args.file).read_text(encoding="utf-8").split("\n\n") if t.strip()]
    else:
        texts = [line for line in sys.stdin.read().split("\n") if line.strip()]
    if not texts:
        _say("error: no text given; pass --text, --file, or pipe text on stdin")
        return 1

    _say(f"loading {artifact} ({manifest.source_id}, budget {manifest.budget}, "
         f"axis {manifest.max_bytes})")

    if args.onnx:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            _say("error: --onnx needs onnxruntime (uv sync --extra onnx)")
            return 1
        session = ort.InferenceSession(
            str(artifact / "artifact.onnx"), providers=["CPUExecutionProvider"]
        )
        for text in texts:
            rows = _onnx_inputs(manifest, artifact, text, args)
            out = session.run(None, rows)[0]
            print(json.dumps({"text": text, "embedding": np.asarray(out[0]).tolist()}))
        return 0

    embedder = ByteifiedEmbedder.load(artifact)
    try:
        embeddings = embedder.encode_text(texts)
    except ValueError as exc:
        _say(f"error: {exc}")
        return 1

    for text, vector in zip(texts, embeddings):
        payload = {
            "text": text,
            "bytes": len(embedder.inputs_for_text(text)[0][0]),
            "embedding": [round(float(x), 6) for x in vector],
        }
        print(json.dumps(payload))

    if len(texts) > 1 and args.similarity:
        sim = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        )
        _say()
        _say("pairwise cosine (centred is meaningless for a single document; this is raw):")
        for row in sim.tolist():
            _say("  " + "  ".join(f"{v:6.3f}" for v in row))
    return 0


def _onnx_inputs(manifest, artifact: Path, text: str, args: argparse.Namespace) -> dict:
    """Build the graph's three inputs with no tokenizer, from the artifact alone."""
    import numpy as np

    from .expand import ByteTrie, expand_byte_ids
    from .manifest import read_trie

    rule = manifest.stream_rule()
    trie = ByteTrie(read_trie(artifact / manifest.trie.path))
    byte_ids = rule.build(text, args.max_bytes or manifest.max_bytes)
    expanded = expand_byte_ids(byte_ids, trie, pseudo_byte_to_token=rule.pseudo_bytes)
    width = manifest.max_bytes
    if len(byte_ids) > width:
        raise ValueError(
            f"text needs {len(byte_ids)} bytes but the artifact's byte axis is {width}. "
            f"This artifact was built with --max-bytes {width} and cannot take longer "
            f"input; rebuild it with a larger axis rather than have text truncated."
        )

    def pad(values, fill):
        return np.pad(
            np.array([values], dtype=np.int64),
            ((0, 0), (0, width - len(values))),
            constant_values=fill,
        )

    # The mask is the stream's own length. A zero *value* is a real byte -- NUL occurs in
    # real text -- so it cannot be used to infer which positions are padding.
    return {
        "byte_ids": pad(byte_ids, 0),
        "expanded_ids": pad(expanded, -1),
        "byte_mask": pad([1] * len(byte_ids), 0),
    }


# -- inspect ---------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    from .manifest import Manifest

    try:
        manifest = Manifest.load(Path(args.artifact))
    except Exception as exc:  # noqa: BLE001
        _say(f"error: {exc}")
        return 1
    _say(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


# -- levels ----------------------------------------------------------------


def cmd_levels(args: argparse.Namespace) -> int:
    from .gates import published_source_ids
    from .levels import LEVELS

    _say("levels")
    for name in sorted(LEVELS, key=lambda n: LEVELS[n].index):
        _say()
        _say(LEVELS[name].describe())
    _say()
    _say("source models with published fidelity thresholds:")
    for model_id in published_source_ids():
        _say(f"  {model_id}")
    _say()
    _say("Stage 2 is not a level: it trains the global model away from the source model,")
    _say("which is exactly what the fidelity guarantee forbids. It stays a research path.")
    return 0


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byteify",
        description="Byteify a subword embedding model into a byte-native serving artifact.",
    )
    parser.add_argument("--version", action="version", version=f"byteify {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build",
        help="byteify a source model into a serving artifact at a chosen level",
        description=(
            "Train the byte interface, gate it, and write an artifact. A failed gate "
            "writes nothing."
        ),
    )
    build.add_argument("--model", required=True,
                       help="HuggingFace model id or a local directory")
    build.add_argument("--level", default="faithful", help="faithful | adapted | merged")
    build.add_argument("--data", nargs="+", required=True,
                       help="reference data: .txt, .json or .jsonl")
    build.add_argument("--judgements", default=None,
                       help="optional relevance judgements: .json or .jsonl")
    build.add_argument("--out", required=True, help="artifact directory to write")
    build.add_argument("--steps", type=int, default=2000)
    build.add_argument("--batch-size", type=int, default=4)
    build.add_argument("--max-bytes", type=int, default=512,
                       help="fixed byte axis; longer inputs fail rather than truncate")
    build.add_argument("--patch-budget", type=int, default=None,
                       help="fixed patch axis (default: fitted to --max-bytes). Surplus "
                            "slots are masked; too few merges patches and costs fidelity")
    build.add_argument("--local-layers", type=int, default=1)
    build.add_argument("--local-ffn", type=int, default=1024)
    build.add_argument("--local-window", type=int, default=128)
    build.add_argument("--lr", type=float, default=3e-4)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--device", default=None, help="cpu | mps | cuda")
    build.add_argument("--delta-from", default=None,
                       help="level merged: a post-trained variant of the source model")
    build.add_argument("--dry-run", action="store_true",
                       help="print the plan and the thresholds, train nothing")
    build.add_argument("--quiet", action="store_true")
    build.set_defaults(func=cmd_build)

    encode = sub.add_parser(
        "encode",
        help="embed text with an artifact, using its manifest alone",
        description=(
            "Loads an artifact through its manifest and embeds text. No source tokenizer "
            "is loaded, and none is needed."
        ),
    )
    encode.add_argument("artifact", help="artifact directory")
    group = encode.add_mutually_exclusive_group()
    group.add_argument("--text", help="a single string")
    group.add_argument("--file", help="a file; paragraphs are split on blank lines")
    encode.add_argument("--onnx", action="store_true", help="run the exported graph")
    encode.add_argument("--max-bytes", type=int, default=None,
                        help="override the artifact's byte axis")
    encode.add_argument("--similarity", action="store_true",
                        help="also print the pairwise cosine of the batch")
    encode.set_defaults(func=cmd_encode)

    inspect = sub.add_parser("inspect", help="print an artifact's manifest")
    inspect.add_argument("artifact")
    inspect.set_defaults(func=cmd_inspect)

    levels = sub.add_parser("levels", help="what each level guarantees, and for which models")
    levels.set_defaults(func=cmd_levels)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())