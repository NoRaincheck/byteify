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

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .bytestream import rule_for_family
from .gates import (
    GateReport,
    Thresholds,
    check_coverage,
    check_exported_graph,
    check_fidelity,
    check_lossless_byte_stream,
    check_retrieval,
    require_retrieval_floor,
    resolve_thresholds,
)
from .levels import declare, resolve_level
from .manifest import (
    GateResults,
    Manifest,
    content_digest,
    export_artifact,
    measure_trie_coverage,
    recipe_digest,
)

#: Bolmo's compression ratio on English prose. Used to fit a patch budget to a byte axis;
#: see `_fit_patch_budget`.
TARGET_BYTES_PER_PATCH = 4.0

#: Held-out texts the fidelity gate grades on. Sixteen is not enough: the centred cosine of
#: the same Stage 1 checkpoint moved between 0.66 and 0.89 across two sixteen-text slices of
#: the same corpus, so a gate that size reports the slice it drew rather than the artifact.
#: Sixty-four holds to about 0.01 and costs a couple of seconds.
GATE_TEXTS = 64


def build_artifact(
    *,
    model: str,
    level: str,
    output_dir: Path,
    data: list[str] | None = None,
    texts: list[str] | None = None,
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
    thresholds: Thresholds | str | Path | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
) -> Manifest:
    """Produce a byteified artifact, or raise. Never returns a failed one.

    This is the library entry point as much as it is the body of ``byteify build``: it
    takes a model id or local directory, returns the written :class:`Manifest`, and is
    silent unless given ``log=``. Nothing about it requires the command line.

    ``data`` is a list of files to read reference text from and ``texts`` is reference
    text already in memory; exactly one is required, because inferring which a caller
    meant from a list of strings that might be either would silently train on paths.

    ``thresholds`` is ``None`` for the model's published floors, or a :class:`Thresholds`
    / path to one measured for *this* model. Supplying them is what makes a source model
    with no published entry byteifiable at all; see :func:`byteify.gates.resolve_thresholds`
    for why borrowing another model's floors stays impossible either way.
    """
    import torch

    from .alignment import discover_special_bytes
    from .data import truncate_to_bytes
    from .hub import resolve_source_model, resolve_source_tokenizer
    from .inference import ByteifiedEmbedder
    from .sentence_transformers import read_pipeline
    from .source import (
        HuggingFaceSourceEncoder,
        LocalEncoderSpec,
        build_byteified_encoder,
        load_encoder_state,
        resolve_source_encoder,
    )
    from .stitching import encode_texts
    from .training import (
        Stage1Config,
        load_judgements,
        load_reference_texts,
        save_checkpoint,
        train_stage1,
    )
    from .vocab import resolve_token_bytes

    if (data is None) == (texts is None):
        raise ValueError(
            "give exactly one of data= (files to read reference text from) or texts= "
            "(reference text already in memory)"
        )

    def say(message: str = "") -> None:
        if log is not None:
            log(message)
        elif not quiet:
            print(message, flush=True)

    started = time.perf_counter()
    chosen = resolve_level(level)

    # -- 1. refuse early ------------------------------------------------------
    # Thresholds first: resolving them reads one packaged file, so an unknown model is
    # refused in milliseconds with no network call -- the contract that "an unknown model
    # is refused here, in seconds, rather than after hours of training" rests on. The
    # pipeline is read next because a repository that declares one byteify cannot
    # reproduce must also be refused before any training is scheduled.
    thresholds, thresholds_origin = resolve_thresholds(model, thresholds)
    pipeline = read_pipeline(model)
    reference: list[str] = (
        list(texts) if texts is not None else load_reference_texts(data, limit=4000 + 256)
    )
    if len(reference) < 64:
        raise ValueError(
            f"reference data yielded {len(reference)} usable texts; at least 64 are needed to "
            "hold some out for evaluation"
        )
    graded = load_judgements(judgements)
    if chosen.needs_judgements and graded is None:
        say(
            f"note: no --judgements given, so level `{chosen.name}` will not gate on "
            "retrieval. Its other guarantees are unaffected."
        )
    if graded is not None:
        # Refused here, in seconds, rather than after training: the whole point of
        # supplying judgements is that the retrieval check can fail, and it cannot fail
        # against a floor nobody published.
        require_retrieval_floor(model, thresholds)
        say(f"  judgements  {len(graded)} decisions from {judgements}, "
            f"chance recall@1 {graded.chance_recall_at_1:.3f}")
    if chosen.index >= 3 and delta_from is None:
        raise ValueError(
            f"level `{chosen.name}` needs --delta-from: a post-trained variant of "
            f"{model} whose task delta is merged into the transformer body"
        )

    say(f"byteify {chosen.name}: {model}")
    say(f"  thresholds  cos_emb_c >= {thresholds.cos_emb_c}, "
        f"sim_corr >= {thresholds.sim_corr}, boundary_acc >= {thresholds.boundary_acc}")
    if thresholds_origin != "published":
        # Said out loud because the artifact will record it too, and a reader should never
        # have to discover from a manifest that the floors under a build were not
        # published ones.
        say(f"              {thresholds_origin}: supplied for {model} rather than "
            f"published; the manifest records that")
    say(f"  pipeline    {pipeline.describe()}")
    say(f"  reference   {len(reference)} texts from "
        f"{', '.join(data) if data else 'the caller'}")
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

    # Everything that determined this build, in one object. Recorded so two builds can be
    # compared on what they were *asked* for -- a digest of the reference data included,
    # because "same recipe, different corpus" is not the same artifact and a recipe that
    # could not tell them apart would be a recipe that cannot be checked.
    recipe = {
        "model": model,
        "level": chosen.name,
        "seed": seed,
        "device": device,
        "steps": steps,
        "batch_size": batch_size,
        "max_bytes": max_bytes,
        "patch_budget": patch_budget,
        "local_layers": local_layers,
        "local_ffn": local_ffn,
        "local_window": local_window,
        "lr": lr,
        "pooling": pipeline.pooling,
        "normalize": pipeline.normalize,
        "thresholds": {
            "cos_emb_c": thresholds.cos_emb_c,
            "sim_corr": thresholds.sim_corr,
            "boundary_acc": thresholds.boundary_acc,
            "onnx_max_abs_diff": thresholds.onnx_max_abs_diff,
            "trie_coverage": thresholds.trie_coverage,
            "retrieval_recall_at_1": thresholds.retrieval_recall_at_1,
        },
        "thresholds_origin": thresholds_origin,
        "reference_digest": texts_digest(reference),
        "judgements_digest": path_digest(judgements),
        "delta_from": delta_from,
    }
    say(f"  recipe      {recipe_digest(recipe)[:16]}... over {len(recipe)} inputs")

    # Weights and vocabulary resolve through the same route, for the same reason: a build
    # that read the encoder from one layout and the vocabulary from another would train a
    # byte interface against a byte stream its own tokenizer does not produce.
    tokenizer = resolve_source_tokenizer(model)
    resolved = resolve_source_model(model)
    source = HuggingFaceSourceEncoder(resolved.model, model_id=model, layout=resolved.layout)
    if resolved.layout != "checkpoint":
        say(f"  source layout: {resolved.layout} ({resolved.module_path})")
    if resolved.dropped:
        say(f"  {len(resolved.dropped)} checkpoint tensor(s) are not encoder weights and "
            f"were not loaded: {sorted(resolved.dropped)[:4]}")
    source = source.to(device)
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
    result = train_stage1(
        encoder, source, tokenizer, reference, cfg, pooling=pipeline.pooling, log=say
    )

    checkpoint = output_dir.parent / "checkpoint.pt"
    save_checkpoint(checkpoint, result.state_dict, {**result.config, **spec.to_config(), "model": model})

    if chosen.index >= 2:
        from .contrastive import train_contrastive_on_corpus

        say(f"\nlevel {chosen.index}: contrastive on the supplied corpus")
        result.state_dict = train_contrastive_on_corpus(
            encoder, source, tokenizer, reference,
            steps=max(steps // 4, 50), seed=seed, device=device,
            pooling=pipeline.pooling, log=say,
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
    # One slice, two uses: trie coverage and the fidelity gate grade the same held-out
    # texts, so the count is written once. Two literals that have to agree, and a gate that
    # quietly graded fewer texts than it promised, is what that invites.
    coverage_texts = [truncate_to_bytes(t, max_bytes) for t in reference[:GATE_TEXTS]]
    coverage = measure_trie_coverage(token_bytes, coverage_texts)
    gate_texts = coverage_texts
    check_lossless_byte_stream(report, tokenizer=tokenizer, rule=rule, texts=gate_texts)

    embedder = ByteifiedEmbedder(
        encoder, source, n_patches=min(patch_budget, max_bytes),
        pooling=pipeline.pooling, normalize=pipeline.normalize,
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
        report, embedder=embedder, source=source, batches=gate_batches,
        thresholds=thresholds, pooling=pipeline.pooling,
    )
    check_coverage(report, coverage, thresholds)

    retrieval = None
    if chosen.index >= 2 and graded is not None:
        # Level 2's own guarantee, on texts the artifact was not fitted on: the
        # judgements carry their queries and candidate pools, and Stage 1 and the
        # contrastive stage never saw them unless the caller passed them as --data too.
        # Grading on the training corpus would measure memorisation and call it retrieval.
        say(f"\nlevel 2: retrieval over {len(graded)} supplied judgements")
        retrieval = check_retrieval(
            report,
            embed=lambda texts: embedder.encode(tokenizer, texts, max_bytes=max_bytes),
            judgements=graded,
            thresholds=thresholds,
            batch_size=batch_size,
        )
        say(f"  decisions {retrieval['n_decisions']}, "
            f"recall@1 {retrieval['recall@1']:.3f} against chance "
            f"{retrieval['chance_recall_at_1']:.3f}, MRR {retrieval['mrr']:.3f}")

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

    # The graph is traced at, and then verified against, the same inputs. Built here and
    # moved onto the model's device: `export_onnx` would otherwise build its own example on
    # CPU, and the dynamo exporter rejects a graph with tensors on two devices.
    graph_inputs = tuple(
        tensor.to(device)
        for tensor in _example_inputs(rule, trie, coverage_texts[:batch_size], max_bytes)
    )

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
        example_batch=graph_inputs,
        pooling=pipeline.pooling,
        normalize=pipeline.normalize,
        source_pipeline=pipeline.as_manifest(),
        recipe=recipe,
    )

    diff = check_exported_graph(
        report, graph_path=output_dir / "artifact.onnx", inputs=graph_inputs,
        eager=_eager(embedder, graph_inputs), thresholds=thresholds,
    )
    if not report.passed:
        # Remove the whole directory: a graph without a passing gate is not an artifact.
        shutil.rmtree(output_dir)
        report.raise_if_failed()

    # The manifest records what the artifact promised and whether it held. Built through
    # `GateResults` rather than merged as a loose dict, because a key that is not a declared
    # field is a key that is silently dropped on the way to disk.
    recorded = GateResults.from_dict({
        **report.record(),
        **declare(chosen, thresholds, report.extra.get("fidelity", metrics)),
        "thresholds_origin": thresholds_origin,
        "fidelity": report.extra.get("fidelity", {}),
        "byte_stream_mismatches": report.extra.get("byte_stream_mismatches"),
        "retrieval": None if retrieval is None else retrieval["recall@1"],
        "retrieval_detail": retrieval or {},
    })
    # Taken here, after every content file exists and before the manifest -- which is
    # excluded from it -- is written. This is the determinism claim in a form a reader can
    # re-check with `verify_artifact`.
    digest = content_digest(output_dir)
    manifest = Manifest.from_dict({
        **manifest.to_dict(),
        "gate": recorded.to_dict(),
        "content_digest": digest,
    })
    manifest.write_to(output_dir / "manifest.json")

    checkpoint.unlink(missing_ok=True)
    say(f"  onnx vs eager: max |diff| {diff:.2e}")
    say(f"  content      {len(digest)} files hashed; verify with byteify.verify_artifact")
    say(f"\ndone in {time.perf_counter() - started:.0f}s -> {output_dir}")
    return manifest


def texts_digest(texts: list[str]) -> str:
    """sha256 of reference text in order, as the data half of a recipe.

    Ordered and hashed whole rather than counted: two corpora of 4000 texts are not the
    same corpus, and a recipe that recorded only a length could not tell them apart.
    """
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def path_digest(path: str | None) -> str:
    """sha256 of a supplied file, or empty when there is none.

    Same reasoning as :func:`texts_digest` for judgements and deltas: the file's *content*
    is what a build used, and recording only its name would let an edited judgements file
    keep the recipe it no longer matches.
    """
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


def _eager(embedder, inputs):
    """The eager result the exported graph is compared against."""
    import torch

    with torch.no_grad():
        return embedder(*inputs)


def _example_inputs(rule, trie, texts: list[str], width: int):
    """One padded batch of the graph's three inputs, built with no tokenizer.

    The byte mask comes from each stream's own length rather than from the byte values. A
    zero *value* is a real byte -- NUL appears in binary-ish text -- so inferring the mask
    from ``byte_ids != 0`` would silently drop real positions, and the gate would then be
    measuring a graph fed a corrupted mask.
    """
    import torch
    import torch.nn.functional as F

    from .expand import expand_byte_ids

    rows = []
    for text in texts:
        byte_ids = rule.build(text, width)
        expanded = expand_byte_ids(
            byte_ids, trie, pseudo_byte_to_token=rule.pseudo_bytes
        )
        rows.append((byte_ids, expanded))

    byte_ids = torch.stack(
        [F.pad(torch.tensor(ids), (0, width - len(ids))) for ids, _ in rows]
    )
    expanded = torch.stack(
        [F.pad(torch.tensor(exp), (0, width - len(exp)), value=-1) for _, exp in rows]
    )
    mask = torch.stack(
        [F.pad(torch.ones(len(ids), dtype=torch.long), (0, width - len(ids))) for ids, _ in rows]
    )
    return byte_ids, expanded, mask