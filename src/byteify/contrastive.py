"""Contrastive training for the byteified embedder.

Stage 1 recovers the source model's *behaviour* and Stage 2 teaches it byte-level
competence, but neither optimises retrieval: both target representations, not the
similarity structure a retriever is judged on. This module supplies the missing third
stage -- the objective that actually makes the output useful as an embedding.

Design notes
------------
**Symmetric InfoNCE with in-batch negatives.** Given an anchor and a positive for each of
``B`` examples, the score matrix is the scaled dot product between the two views'
normalised embeddings; the loss is the cross-entropy of the diagonal in both directions.
In-batch negatives are what makes this cheap: a batch of ``B`` pairs yields ``B - 1``
negatives for free, which matters here because every negative is a *byte-level forward
pass*.

**Temperature is learned**, bounded away from zero, as in SimCSE. A fixed temperature is
a real hyperparameter at this scale; a learned one adapts to how concentrated the
embedding space is.

**The embedding is pooled over the budgeted patch axis**, exactly as at inference, so
training optimises the representation that will actually be used rather than a
train/inference mismatch.

**Pairs come from wherever they can be sourced.** :func:`mine_pairs` builds them locally
by embedding a corpus with the *source* model, so no external dataset is required and the
positives are, by construction, near-duplicates of the anchors in the teacher's metric.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProjectionHead(nn.Module):
    """Optional MLP projection, as in SimCSE.

    Retrieval is evaluated on the projected vector by default, matching that paper. Set
    ``dim=0`` to train and evaluate the raw pooled embedding instead, which is the
    convention used by E5/BGE-style models and by this project's fidelity metrics.
    """

    def __init__(self, d_model: int, dim: int = 0, hidden: int = 2048) -> None:
        super().__init__()
        self.enabled = dim > 0
        if self.enabled:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) if self.enabled else x


@dataclass
class ContrastiveLoss:
    total: Tensor
    anchor: Tensor
    positive: Tensor
    accuracy: Tensor
    temperature: float


def infonce(
    anchor: Tensor,
    positive: Tensor,
    *,
    temperature: Tensor | float = 0.05,
) -> ContrastiveLoss:
    """Symmetric InfoNCE with in-batch negatives.

    anchor, positive: [B, D], unnormalised. Both are L2-normalised internally.

    The score matrix is ``a @ p.T`` -- anchor *i* against positive *j*. The target is the
    diagonal, so ``i == j`` is the true pair and every other column is a negative.
    """
    a = F.normalize(anchor.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    logits = a @ p.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    total = 0.5 * (loss_a + loss_p)

    with torch.no_grad():
        # Fraction of examples whose true positive is the nearest of the batch.
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    # Detach: this is a logged statistic, and a grad-tracking scalar forces callers to
    # remember `.detach()` at every logging site.
    temp = float(temperature.detach()) if isinstance(temperature, Tensor) else float(temperature)
    return ContrastiveLoss(
        total=total, anchor=loss_a.detach(), positive=loss_p.detach(),
        accuracy=accuracy, temperature=temp,
    )


class ContrastiveHead(nn.Module):
    """Projection head plus a learned, bounded temperature."""

    def __init__(self, d_model: int, proj_dim: int = 0, hidden: int = 2048, init_temp: float = 0.05) -> None:
        super().__init__()
        self.projection = ProjectionHead(d_model, proj_dim, hidden)
        # Parameterised in log space and clamped so the softmax cannot collapse.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))

    def temperature(self) -> Tensor:
        return self.log_temperature.clamp(-4.0, 4.0).exp()

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        return self.projection(pooled), self.temperature()


@dataclass
class PairBatch:
    anchors: list[str]
    positives: list[str]


def mine_pairs(
    texts: Sequence[str],
    embed_fn: Callable[[Sequence[str]], Tensor],
    *,
    pool_size: int = 64,
    n_per_anchor: int = 1,
    temperature: float = 0.05,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Mine positive pairs by nearest neighbours in the *source* model's space.

    ``embed_fn`` maps a list of texts to normalised embeddings. Positives are the
    ``n_per_anchor`` nearest neighbours of each anchor, excluding itself. Softmax
    sampling over similarity temperature sharpens selection toward the hardest positives,
    which are the informative ones.

    This needs no external dataset and no byte-level forward passes: the source model is
    already available and its metric is a sensible notion of "similar".
    """
    if len(texts) < 2:
        return []
    rng = random.Random(seed)
    embeddings = F.normalize(embed_fn(list(texts)).float(), dim=-1)

    pairs: list[tuple[str, str]] = []
    for start in range(0, len(texts), pool_size):
        window = list(range(start, min(start + pool_size, len(texts))))
        block = embeddings[window]
        sim = block @ block.T / temperature
        sim.fill_diagonal_(float("-inf"))
        weights = torch.softmax(sim, dim=-1).cpu()
        for row, anchor_idx in enumerate(window):
            picks = torch.multinomial(weights[row], min(n_per_anchor, len(window) - 1))
            for pick in picks.tolist():
                pairs.append((texts[anchor_idx], texts[window[pick]]))
    rng.shuffle(pairs)
    return pairs


def pair_batches(
    pairs: Sequence[tuple[str, str]], batch_size: int, seed: int = 0
) -> Iterable[list[tuple[str, str]]]:
    """Shuffled fixed-size batches over pairs."""
    order = list(pairs)
    random.Random(seed).shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
        yield order[i : i + batch_size]


def train_contrastive_on_corpus(
    encoder,
    source,
    tokenizer,
    texts: Sequence[str],
    *,
    steps: int = 500,
    batch_size: int = 8,
    lr: float = 1e-4,
    temperature: float = 0.05,
    max_bytes: int = 512,
    seed: int = 0,
    device: str = "cpu",
    pooling: str = "mean",
    log=lambda message: None,
) -> dict:
    """Fit the byte interface to the similarity structure of a corpus.

    This is the objective Stage 1 and Stage 2 do not optimise. Stage 1 imitates a source
    model that was never trained on the practitioner's domain; this fits the domain, which
    is what level 2 is for.

    Positives are mined in the *source* model's space, so the pairs reflect real semantic
    similarity rather than whatever the byte interface currently believes. Returns the
    encoder's new state dict.

    ``pooling`` keeps both spaces on the source's declared pooling mode: pairs are mined
    from pooled source embeddings and the student is trained against pooled student ones,
    so a CLS-pooled model mines and grades CLS pools rather than mean ones.
    """
    import torch
    import torch.nn.functional as F

    from .data import truncate_to_bytes
    from .fidelity import _budgeted_patch_reps, pool_final
    from .inference import ByteifiedEmbedder
    from .stitching import encode_texts

    if len(texts) < 8:
        raise ValueError(f"contrastive training needs at least 8 texts; got {len(texts)}")

    torch.manual_seed(seed)
    encoder.to(device).train()
    embedder = ByteifiedEmbedder(
        encoder, source, n_patches=min(64, max_bytes), pooling=pooling
    ).to(device)

    def source_embeddings(batch_texts: list[str]) -> torch.Tensor:
        rows = encode_texts(
            [truncate_to_bytes(t, max_bytes) for t in batch_texts],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        with torch.no_grad():
            return pool_final(
                source.representation_at(rows.input_ids, rows.attention_mask, source.n_layers),
                rows.attention_mask.bool(),
                source.final_norm,
                mode=pooling,
            )

    pairs = mine_pairs(
        texts,
        source_embeddings,
        pool_size=min(128, len(texts)),
        n_per_anchor=1,
        temperature=temperature,
        seed=seed,
    )
    if not pairs:
        raise ValueError("no positive pairs could be mined from the reference corpus")
    log(f"  mined {len(pairs)} positive pairs from {len(texts)} texts")

    head = ContrastiveHead(encoder.embed.byte.weight.shape[1]).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    # The head learns the temperature, so it is the head's own value rather than the
    # argument, and it comes back with every projection. Unpacking it here is what makes
    # the learned temperature actually train.
    def project(rows) -> Tensor:
        return head(embed(rows))[0]

    def embed(batch_rows) -> torch.Tensor:
        reps, mask, _ = _budgeted_patch_reps(embedder, batch_rows)
        hidden = source.forward_layers(reps, mask, n_layers=source.n_layers)
        # Normalised regardless of the artifact's own normalize flag: this is InfoNCE,
        # which needs unit vectors to have the temperature it learns any meaning. The
        # artifact's output space is a serving decision and does not apply here.
        return F.normalize(pool_final(hidden, mask, source.final_norm, mode=pooling), dim=-1)

    for done, batch in enumerate(pair_batches(pairs, batch_size, seed=seed), start=1):
        if done > steps:
            break
        # Two batches, not one sliced: `encode_texts` returns a ByteBatch, which is not a
        # sequence of rows. Indexing it raised TypeError, which meant level 2 -- the level
        # this module exists for -- had never run end to end.
        anchors = encode_texts(
            [truncate_to_bytes(pair[0], max_bytes) for pair in batch],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        positives = encode_texts(
            [truncate_to_bytes(pair[1], max_bytes) for pair in batch],
            tokenizer,
            add_special_tokens=True,
            device=device,
        )
        loss = infonce(
            project(anchors), project(positives),
            temperature=head.temperature(),
        )
        opt.zero_grad(set_to_none=True)
        # `infonce` returns a ContrastiveLoss, not a tensor: the anchor and positive halves
        # and the in-batch accuracy are all reported, and calling `.backward()` on the
        # record is the error a caller makes when they have not read the signature.
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if done % max(1, steps // 10) == 0:
            log(f"  step {done:>6}  infonce {loss.total.item():.4f}  "
                f"accuracy {loss.accuracy.item():.3f}  temperature {loss.temperature:.4f}")

    encoder.eval()
    return encoder.state_dict()


@torch.no_grad()
def retrieval_over_judgements(
    embed,
    judgements,
    *,
    batch_size: int = 8,
    max_bytes: int = 512,
) -> dict[str, float]:
    """Retrieval metrics over a judgement set whose queries each have their own pool.

    ``embed(texts) -> [n, d]``. Each query is ranked only against its own candidates,
    which is the decision the judgements were authored to describe. Chance is reported
    alongside, because a mean over pools of different sizes has no single chance level and
    a number without one cannot be read.
    """
    totals: dict[str, float] = {}
    queries = list(judgements.queries)

    # Candidates are pooled by text before embedding: the same candidate string is
    # embedded once and reused, and the number of byte-level forward passes is bounded by
    # the distinct strings rather than by decisions times candidates.
    unique: dict[str, int] = {}
    for pool in judgements.candidates:
        for candidate in pool:
            unique.setdefault(candidate, len(unique))
    candidate_texts = list(unique)

    def embed_all(texts: list[str]) -> Tensor:
        rows = []
        for start in range(0, len(texts), batch_size):
            rows.append(embed(texts[start : start + batch_size]))
        return torch.cat(rows) if rows else torch.empty(0)

    query_vectors = embed_all(queries)
    candidate_vectors = embed_all(candidate_texts)

    for index, pool in enumerate(judgements.candidates):
        columns = [unique[candidate] for candidate in pool]
        metrics = retrieval_metrics(
            query_vectors[index : index + 1],
            candidate_vectors[columns],
            {0: judgements.relevant[index]},
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value

    n = max(len(judgements), 1)
    return {
        **{key: value / n for key, value in totals.items()},
        "n_decisions": len(judgements),
        "chance_recall_at_1": judgements.chance_recall_at_1,
    }


def retrieval_metrics(
    query_embeddings: Tensor,
    doc_embeddings: Tensor,
    relevant: dict[int, list[int]],
) -> dict[str, float]:
    """Recall@k, MRR and nDCG for a query -> document similarity matrix.

    ``relevant`` maps a query index to the indices of documents that answer it.
    Anisotropy matters here as much as anywhere else in this project, so recall is the
    primary number: it is insensitive to the common direction that inflates cosine.
    """
    q = F.normalize(query_embeddings.float(), dim=-1)
    d = F.normalize(doc_embeddings.float(), dim=-1)
    sim = q @ d.T
    ranks = sim.argsort(dim=-1, descending=True)

    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr = 0.0
    ndcg = 0.0
    for qi, gold in relevant.items():
        order = ranks[qi].tolist()
        position = {doc_id: rank for rank, doc_id in enumerate(order)}
        best = min((position[g] for g in gold if g in position), default=len(order))
        for k in recalls:
            recalls[k] += float(best < k)
        mrr += 1.0 / (best + 1)
        gains = [1.0 / math.log2(rank + 2) for rank, g in enumerate(order) if g in gold]
        ideal = [1.0 / math.log2(i + 2) for i in range(len(gold))]
        ndcg += sum(gains) / max(sum(ideal), 1e-9)

    n = max(len(relevant), 1)
    out = {f"recall@{k}": v / n for k, v in recalls.items()}
    out["mrr"] = mrr / n
    out["ndcg"] = ndcg / n
    return out