"""Stage 1 training: teach a byte interface to stand in for a frozen subword encoder.

The global model stays frozen. This is the paper's efficiency trick, and it costs roughly a
third of end-to-end training: the byte interface learns to reproduce the *representations*
the source encoder already produces, so nothing has to be retrained behind it.

The recipe is a function here rather than in a script, so the ``byteify build`` command and
``scripts/train_stage1.py`` cannot drift into training two different things and reporting
progress from one of them.

Held-out fidelity is reported throughout, not just loss. Loss going down does not mean the
byte interface can stand in for the subword model; the questions that matter are whether
the boundary decisions agree and whether the two models' similarity structure agrees. See
:mod:`byteify.fidelity` for what each number means.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .data import BatchSampler, iter_texts, truncate_to_bytes
from .fidelity import measure_fidelity
from .losses import (
    Stage1Weights,
    boundary_loss,
    embedding_loss,
    stitching_loss,
)
from .stitching import encode_texts


@dataclass
class Stage1Config:
    steps: int = 2000
    batch_size: int = 4
    max_bytes: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup: int = 100
    stitch_depth: int = -1  # -1 = full depth
    weights: Stage1Weights = field(default_factory=lambda: Stage1Weights(4.0, 1.0, 1.0))
    eval_every: int = 500
    eval_batches: int = 4
    eval_texts: int = 256
    n_train: int = 4000
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 50
    checkpoint_every: int = 500


@dataclass
class Stage1Result:
    state_dict: dict
    config: dict
    history: list[dict]
    metrics: dict


def _lr_at(step: int, cfg: Stage1Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def _fidelity(encoder, source, batches) -> dict[str, float]:
    """Held-out fidelity of a bare ``ByteifiedEncoder``, under oracle boundaries.

    Oracle, not budgeted: Stage 1 pools with the true subword boundaries so the patch axis
    stays aligned with the teacher's token axis, so that is what the reported number has to
    describe. The gate measures budgeted patching instead, because that is what the
    exported graph does. Two different measurements of two different things, reported as
    such.
    """
    from .fidelity import _oracle_patch_reps

    was_training = encoder.training
    encoder.eval()
    metrics = measure_fidelity(lambda batch: _oracle_patch_reps(encoder, batch), source, batches)
    if was_training:
        encoder.train()
    return metrics


def train_stage1(
    encoder,
    source,
    tokenizer,
    texts: list[str],
    cfg: Stage1Config,
    *,
    log=lambda message: None,
) -> Stage1Result:
    """Run Stage 1 and return the trained state dict plus its held-out metrics."""
    torch.manual_seed(cfg.seed)
    device = cfg.device

    if len(texts) < 64:
        raise ValueError(
            f"reference data has {len(texts)} texts; Stage 1 needs at least 64 to hold "
            "some out for evaluation. Run scripts/prepare_corpus.py, or pass more --data."
        )

    # Hold out at most a quarter, and never the whole set. Taking a flat `eval_texts` slice
    # silently consumed every text when the corpus was smaller than that default, leaving
    # the sampler with nothing -- and `BatchSampler.epoch()` is an unbounded generator, so
    # training spun forever rather than reporting that it had no data.
    n_eval = min(cfg.eval_texts, max(1, len(texts) // 4))
    eval_texts, train_texts = texts[:n_eval], texts[n_eval:]
    if not train_texts:
        raise ValueError(
            f"reference data has {len(texts)} texts, all of which were held out for "
            "evaluation; lower --eval-texts or supply more --data"
        )
    sampler = BatchSampler(
        train_texts,
        tokenizer,
        batch_size=cfg.batch_size,
        max_bytes=cfg.max_bytes,
        add_special_tokens=True,
        seed=cfg.seed,
    )
    # Truncation is the same rule the sampler uses. Calling encode_texts on untruncated
    # documents made the evaluation byte axis as long as the longest document, which blew
    # the attention guard and produced a meaningless number.
    eval_batches = [
        encode_texts(
            [truncate_to_bytes(t, cfg.max_bytes) for t in
             eval_texts[i : i + cfg.batch_size]],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        for i in range(0, min(n_eval, cfg.eval_batches * cfg.batch_size), cfg.batch_size)
    ]

    depth = source.n_layers if cfg.stitch_depth < 0 else cfg.stitch_depth
    params = [p for p in encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    log(
        f"stage 1: {len(train_texts)} train / {len(eval_texts)} held-out texts, "
        f"stitching at depth n={depth}, lambda_B={cfg.weights.boundary} "
        f"lambda_E={cfg.weights.stitching} lambda_emb={cfg.weights.embedding}"
    )

    history: list[dict] = []
    step = 0
    started = time.perf_counter()
    encoder.train()
    while step < cfg.steps:
        for batch in sampler.epoch():
            if step >= cfg.steps:
                break
            batch = batch.to(device)
            for group in opt.param_groups:
                group["lr"] = _lr_at(step, cfg)

            out = encoder(
                batch.byte_ids, batch.expanded_ids, batch.byte_mask,
                boundaries=batch.boundaries,
            )
            teacher = source.representation_at(batch.input_ids, batch.attention_mask, depth)
            student = source.forward_layers(out.patch_reps, out.patch_mask, n_layers=depth)

            l_b = boundary_loss(out.boundary_logits, batch.boundaries, batch.byte_mask)
            l_e = stitching_loss(student, teacher, out.patch_mask)
            l_emb = embedding_loss(
                student, teacher, out.patch_mask, batch.attention_mask.bool(), source.final_norm
            )
            total = cfg.weights.boundary * l_b + cfg.weights.stitching * l_e \
                + cfg.weights.embedding * l_emb

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            if step % cfg.log_every == 0:
                elapsed = time.perf_counter() - started
                log(
                    f"  step {step:>6}  total {total.item():>9.4f}  L_B {l_b.item():>7.4f}  "
                    f"L_E {l_e.item():>8.4f}  L_emb {l_emb.item():>7.4f}  "
                    f"{elapsed / max(step, 1) * 1000:>6.0f} ms/step"
                )
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                metrics = _fidelity(encoder, source, eval_batches)
                history.append({"step": step, **metrics})
                log(
                    f"  step {step:>6}  held-out: sim_corr={metrics['sim_corr']:.3f} "
                    f"cos_emb_c={metrics['cos_emb_c']:.3f} "
                    f"nn_agree={metrics['nn_agree']:.3f}/{metrics['nn_chance']:.3f} "
                    f"boundary={metrics['boundary_acc']:.3f}"
                )

    metrics = history[-1] if history else {}
    return Stage1Result(
        state_dict=encoder.state_dict(),
        config={
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "max_bytes": cfg.max_bytes,
            "lr": cfg.lr,
            "stitch_depth": cfg.stitch_depth,
            "eval_every": cfg.eval_every,
            "seed": cfg.seed,
            "lambda_b": cfg.weights.boundary,
            "lambda_emb": cfg.weights.embedding,
            "lambda_e": cfg.weights.stitching,
        },
        history=history,
        metrics=metrics,
    )


def load_reference_texts(paths: list[str], *, min_chars: int = 64, limit: int | None = None) -> list[str]:
    """Plain text is the floor: documents as .txt, .json or .jsonl.

    No labelled data is required to get a working artifact. Relevance judgements are
    separate and optional.
    """
    return list(iter_texts(paths, min_chars=min_chars, limit=limit))


@dataclass(frozen=True, slots=True)
class Judgements:
    """Relevance judgements as a list of decisions, each with its own candidate pool.

    Candidate-pool-per-query, rather than one global pool of documents, because that is
    what a retrieval evaluation needs to mean. Given a query, which of *these* candidates
    answers it -- that is a decision a practitioner can act on and a model can be graded
    on. Pooling every document in the corpus behind every query measures something else,
    and its chance level depends on how large the corpus happens to be, which makes the
    number impossible to compare between two artifacts fitted to different corpora.

    Accepts two row shapes, both of which have existed here:

    * ``{"query": q, "candidates": [...], "positive": i or text}`` -- a candidate pool.
    * ``{"query": q, "positive": doc}`` -- one document, which is a pool of one. It is
      accepted so an existing two-field file keeps working; every metric over it is
      trivially perfect, which is why it is a convenience and not the format to write.
    """

    queries: list[str] = field(default_factory=list)
    candidates: list[list[str]] = field(default_factory=list)
    #: Candidate indices that answer each query. Indices, not texts, because the same
    #: document can legitimately be relevant to two different queries.
    relevant: list[list[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def candidate_counts(self) -> list[int]:
        return [len(pool) for pool in self.candidates]

    @property
    def chance_recall_at_1(self) -> float:
        """What a random ranker scores, averaged over the decisions' own pool sizes."""
        counts = self.candidate_counts
        return sum(1.0 / max(n, 1) for n in counts) / max(len(counts), 1)


def load_judgements(path: str | None) -> Judgements | None:
    """Read judgements from a ``.json`` or ``.jsonl`` file; ``None`` when none was given.

    A row that names no correct candidate, or names one that is not in its own pool, is
    dropped rather than repaired. The alternative -- treating it as the first candidate,
    or searching the pool for a near match -- would put a number on a label nobody
    supplied, and that number would then gate a build.
    """
    if path is None:
        return None

    raw = Path(path).read_text(encoding="utf-8")
    rows: list[dict] = []
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        if isinstance(data, dict):
            rows = [
                {"query": q, "positive": v["positive"] if isinstance(v, dict) else v}
                for q, v in data.items()
            ]
        elif isinstance(data, list):
            rows = data

    queries: list[str] = []
    pools: list[list[str]] = []
    relevant: list[list[int]] = []
    for row in rows:
        query = row.get("query")
        positive = row.get("positive")
        pool = list(row.get("candidates") or [])
        if not query or positive is None:
            continue
        if not pool:
            pool = [str(positive)]
        if isinstance(positive, int):
            index = positive
            if not 0 <= index < len(pool):
                continue
        else:
            matches = [i for i, candidate in enumerate(pool) if candidate == positive]
            if not matches:
                continue
            index = matches[0]
        queries.append(query)
        pools.append(pool)
        relevant.append([index])

    if not queries:
        raise ValueError(
            f"{path} yielded no usable judgements; each row needs a query and a positive "
            "that appears among that query's candidates"
        )
    return Judgements(queries=queries, candidates=pools, relevant=relevant)


def save_checkpoint(path: Path, state_dict: dict, config: dict, *, step: int | None = None) -> None:
    """Write a checkpoint atomically, so a crash cannot corrupt the only copy."""
    payload = {"encoder": state_dict, "config": config}
    if step is not None:
        payload["step"] = step
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)