"""``byteify build``: one command from a source model to a gated artifact.

The order matters and is the whole design:

1. Resolve the source model and its **published** thresholds. An unknown model is refused
   here, in seconds, rather than after hours of training.
2. Train the byte interface (Stage 1; contrastive for level 2; a task-arithmetic delta for
   level 3).
3. Measure. Gate. **A failed gate writes nothing.**
4. Only then export the graph and write the manifest.

Doing the gates before the export is deliberate. The graph is the expensive, large part of
the artifact, and writing it for a model that failed its own guarantee is how you end up
with directories full of things that look deployable and are not.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .bytestream import rule_for_family
from .gates import (
    GateReport,
    check_coverage,
    check_exported_graph,
    check_fidelity,
    check_lossless_byte_stream,
    thresholds_for,
)
from .levels import declare, resolve_level
from .manifest import Manifest, export_artifact, measure_trie_coverage

#: Bolmo's compression ratio on English prose. Used to fit a patch budget to a byte axis;
#: see `_fit_patch_budget`.
TARGET_BYTES_PER_PATCH = 4.0


def build_artifact(
    *,
    model: str,
    level: str,
    data: list[str],
    output_dir: Path,
    steps: int = 2000,
    batch_size: int = 4,
    max_bytes: int = 512,
    patch_budget: int | None = None,
    local_layers: int = 1,
    local_ffn: int = 1024,
    local_window: int = 128,
    lr: float = 3e-4,
    seed: int = 0,
    device: str | None = None,
    judgements: str | None = None,
    delta_from: str | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> Manifest:
    """Produce a byteified artifact, or raise. Never returns a failed one."""
    import torch
    from transformers import AutoTokenizer

    from .alignment import discover_special_bytes
    from .data import truncate_to_bytes
    from .inference import ByteifiedEmbedder
    from .source import (
        LocalEncoderSpec,
        build_byteified_encoder,
        load_encoder_state,
        resolve_source_encoder,
    )
    from .stitching import encode_texts
    from .training import Stage1Config, load_reference_texts, save_checkpoint, train_stage1
    from .vocab import resolve_token_bytes

    def say(message: str = "") -> None:
        if not quiet:
            print(message, flush=True)

    started = time.perf_counter()
    chosen = resolve_level(level)

    # -- 1. refuse early ------------------------------------------------------
    thresholds = thresholds_for(model)
    texts = load_reference_texts(data, limit=4000 + 256)
    if len(texts) < 64:
        raise ValueError(
            f"reference data yielded {len(texts)} usable texts; at least 64 are needed to "
            "hold some out for evaluation"
        )
    if chosen.needs_judgements and judgements is None:
        say(
            f"note: no --judgements given, so level `{chosen.name}` will not gate on "
            "retrieval. Its other guarantees are unaffected."
        )
    if chosen.index >= 3 and delta_from is None:
        raise ValueError(
            f"level `{chosen.name}` needs --delta-from: a post-trained variant of "
            f"{model} whose task delta is merged into the transformer body"
        )

    say(f"byteify {chosen.name}: {model}")
    say(f"  thresholds  cos_emb_c >= {thresholds.cos_emb_c}, "
        f"sim_corr >= {thresholds.sim_corr}, boundary_acc >= {thresholds.boundary_acc}")
    say(f"  reference   {len(texts)} texts from {', '.join(data)}")
    patch_budget = _fit_patch_budget(patch_budget, max_bytes=max_bytes, say=say)

    estimate = steps * 0.12  # measured on the 16.8M anchor on MPS
    say()
    say("plan:")
    for label, seconds in [
        ("load source model and byte interface", 4),
        (f"stage 1, {steps} steps", estimate),
        (f"export and verify the graph ({patch_budget} patches, {max_bytes} bytes)", 25),
        ("gates", 10),
    ]:
        when = f"~{seconds:.0f}s" if seconds < 90 else f"~{seconds / 60:.1f} min"
        say(f"  {label:<50} {when:>9}")
    say()

    if dry_run:
        say("dry run: nothing was trained or written.")
        return None

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        # A previous failed run may have left files; a previous *successful* one should be
        # raised explicitly, because silently replacing a good artifact is worse than
        # refusing.
        if (output_dir / "manifest.json").exists():
            raise FileExistsError(
                f"{output_dir} already holds an artifact. Pass a different --out, or "
                "remove it. byteify does not overwrite an artifact it cannot verify."
            )
        shutil.rmtree(output_dir)

    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model)
    source = resolve_source_encoder(model).to(device)
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    spec = LocalEncoderSpec(
        layers=local_layers,
        ffn_hidden=local_ffn,
        window=local_window,
        boundary_predictor="mlp",
    )
    encoder = build_byteified_encoder(source, tokenizer, spec).to(device)
    encoder.embed.load_suffix_from(source.embedding_matrix())

    n_local = sum(p.numel() for p in encoder.local_encoder.parameters())
    n_source = sum(p.numel() for p in source.parameters())
    say(f"  byte interface: {spec.layers}x{spec.backend} ffn={spec.ffn_hidden} "
        f"window={spec.window} (+{n_local / 1e6:.2f}M params, "
        f"{n_local / max(n_source, 1) * 100:.1f}% of the source's {n_source / 1e6:.1f}M)")

    # -- 2. train -------------------------------------------------------------
    cfg = Stage1Config(
        steps=steps,
        batch_size=batch_size,
        max_bytes=max_bytes,
        lr=lr,
        seed=seed,
        device=device,
    )
    result = train_stage1(encoder, source, tokenizer, texts, cfg, log=say)

    checkpoint = output_dir.parent / "checkpoint.pt"
    save_checkpoint(checkpoint, result.state_dict, {**result.config, **spec.to_config(), "model": model})

    if chosen.index >= 2:
        from .contrastive import train_contrastive_on_corpus

        say(f"\nlevel {chosen.index}: contrastive on the supplied corpus")
        result.state_dict = train_contrastive_on_corpus(
            encoder, source, tokenizer, texts,
            steps=max(steps // 4, 50), seed=seed, device=device, log=say,
        )
        load_encoder_state(encoder, result.state_dict, source="contrastive stage")

    if chosen.index >= 3:
        from .task_arithmetic import task_arithmetic_merge

        say(f"\nlevel 3: merging the task delta from {delta_from}")
        variant = resolve_source_encoder(delta_from)
        unmapped = task_arithmetic_merge(source, variant, strict=False)
        if unmapped:
            say(f"  warning: {len(unmapped)} source weights had no counterpart in "
                f"{delta_from} and were left alone: {sorted(unmapped)[:5]}")
        else:
            say("  every source weight mapped to a counterpart")

    # -- 3. gate --------------------------------------------------------------
    report = GateReport()
    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    rule = rule_for_family(
        resolution.strategy, n_byte_values=256 + len(specials), special_tokens=specials
    )
    token_bytes = {i: b for i, b in enumerate(resolution.token_bytes) if b}
    coverage_texts = [truncate_to_bytes(t, max_bytes) for t in texts[:64]]
    coverage = measure_trie_coverage(token_bytes, coverage_texts)

    gate_texts = [t for t in coverage_texts[:16]]
    check_lossless_byte_stream(report, tokenizer=tokenizer, rule=rule, texts=gate_texts)

    embedder = ByteifiedEmbedder(
        encoder, source, n_patches=min(patch_budget, max_bytes)
    ).to(device).eval()
    _, trie, _ = _tables(tokenizer)

    gate_batches = [
        encode_texts(
            [truncate_to_bytes(t, max_bytes) for t in gate_texts[i : i + batch_size]],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        for i in range(0, len(gate_texts) - batch_size + 1, batch_size)
    ]
    metrics = check_fidelity(
        report, embedder=embedder, source=source, batches=gate_batches, thresholds=thresholds
    )
    check_coverage(report, coverage, thresholds)

    say()
    say("gates")
    for check in report.checks:
        mark = "ok  " if check.passed else "FAIL"
        say(f"  {mark} {check.name:<24} {check.measured:>10.5g} "
            f"{check.comparison} {check.threshold:.5g}")
    if report.failures:
        report.raise_if_failed()  # raises before anything is written

    # -- 4. export ------------------------------------------------------------
    say()
    say("exporting")
    manifest = export_artifact(
        embedder,
        tokenizer,
        checkpoint=checkpoint,
        output_dir=output_dir,
        budget=patch_budget,
        max_bytes=max_bytes,
        source_revision=getattr(source, "revision", "") or "",
        coverage_texts=coverage_texts,
        gate=None,
    )

    # The numerical gate runs against the graph that was just written, and the result goes
    # into the manifest, so the artifact records what it was checked against.
    graph_inputs = _example_inputs(embedder, rule, trie, coverage_texts[:batch_size], max_bytes)
    with torch.no_grad():
        eager = embedder(*graph_inputs)
    diff = check_exported_graph(
        report, graph_path=output_dir / "artifact.onnx", inputs=graph_inputs,
        eager=eager, thresholds=thresholds,
    )
    if not report.passed:
        # Remove the whole directory: a graph without a passing gate is not an artifact.
        shutil.rmtree(output_dir)
        report.raise_if_failed()

    manifest = Manifest.from_dict(
        {**manifest.to_dict(), "gate": {**report.record(), **declare(
            chosen, thresholds, report.extra.get("fidelity", metrics)
        )}}
    )
    manifest.write_to(output_dir / "manifest.json")

    checkpoint.unlink(missing_ok=True)
    say(f"  onnx vs eager: max |diff| {diff:.2e}")
    say(f"\ndone in {time.perf_counter() - started:.0f}s -> {output_dir}")
    return manifest


def _tables(tokenizer):
    from .stitching import tokenizer_tables

    return tokenizer_tables(tokenizer)


def _fit_patch_budget(
    patch_budget: int | None, *, max_bytes: int, say
) -> int:
    """Choose a patch budget matched to the byte axis, or warn that it is not.

    The patch budget and the byte axis are not independent. Bolmo's compression ratio is
    about 4.3 bytes per patch on English prose, so a 512-byte axis needs roughly 120
    patches; a budget of 64 forces the extra patches to merge and measurably costs
    fidelity -- on the anchor model, centred cosine 0.585 at budget 64 against 0.795 at
    budget 128 for the same checkpoint. Going the other way wastes compute on masked slots,
    and is also measured (0.762 at budget 256).

    So a default of "64" was simply wrong for any axis above about 256 bytes. The budget
    now follows the axis and the data, and a user who set one that does not fit is told.
    """
    if patch_budget is None:
        chosen = max(8, round(max_bytes / TARGET_BYTES_PER_PATCH))
        say(f"  patch budget {chosen} (derived from a {max_bytes}-byte axis at "
            f"{TARGET_BYTES_PER_PATCH} bytes/patch)")
        return chosen

    implied = patch_budget * TARGET_BYTES_PER_PATCH
    if implied < max_bytes * 0.6:
        say()
        say(
            f"  warning: a patch budget of {patch_budget} on a {max_bytes}-byte axis implies "
            f"{implied / max_bytes:.1f} bytes per patch, against a tokenizer ratio of about "
            f"{TARGET_BYTES_PER_PATCH}. The surplus will merge into fewer patches, and the "
            f"embedding is being computed at a budget you did not intend. Either raise "
            f"--patch-budget to about {max(8, round(max_bytes / TARGET_BYTES_PER_PATCH))} or "
            f"lower --max-bytes."
        )
    return patch_budget


def _example_inputs(embedder, rule, trie, texts: list[str], width: int):
    """One padded batch of the graph's three inputs, built with no tokenizer."""
    import torch
    import torch.nn.functional as F

    from .expand import expand_byte_ids

    rows = []
    for text in texts:
        byte_ids = rule.build(text, width)
        rows.append((byte_ids, expand_byte_ids(byte_ids, trie, pseudo_byte_to_token=rule.pseudo_bytes)))
    target = width
    byte_ids = torch.stack(
        [F.pad(torch.tensor(r[0]), (0, target - len(r[0]))) for r in rows]
    )
    expanded = torch.stack(
        [F.pad(torch.tensor(r[1]), (0, target - len(r[1])), value=-1) for r in rows]
    )
    mask = (byte_ids != 0).long()
    mask[:, 0] = 1  # the opening pseudo-byte is a real position, not padding
    return byte_ids, expanded, mask


def _resumable(output_dir: Path) -> bool:
    return False