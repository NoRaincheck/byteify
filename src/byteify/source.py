"""Adapting any HuggingFace encoder into a byteifiable "source model".

Byteification needs three things from the model it is retrofitting:

1. a **d_model** and **n_layers**,
2. the **input embedding matrix**, so that ``T_SubwordSuffix`` can broadcast pretrained
   subword embeddings onto bytes, and
3. the ability to run the **first n layers over supplied representations**, because
   Stage 1's stitching loss compares the student and the teacher *inside* the frozen
   model at the same depth.

Different HF encoder families expose these differently: ModernBERT uses
``embeddings`` + ``layers`` + a fused ``Wqkv`` and two ``layer_types``; BERT-family
models use ``embeddings`` + ``encoder.layer`` and a single attention mask. Rather than
depend on either, ``SourceEncoder`` normalises them behind one interface, so
byteifying a new embedding model is a config change rather than a code change.

Currently supported: ModernBERT (ettin-*, ModernBERT-base) and BERT-family encoders
(BERT, MiniLM, E5, BGE, DeBERTa-v3, RoBERTa). Anything with
``embeddings``/``layers`` or ``embeddings``/``encoder.layer`` and ``config.hidden_size``
should route through :class:`HuggingFaceSourceEncoder`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


class SourceEncoder(nn.Module):
    """The frozen model being retrofit, seen through the interface byteification needs."""

    d_model: int
    n_layers: int
    vocab_size: int

    def embedding_matrix(self) -> Tensor:
        """The pretrained input embedding matrix ``[vocab_size, d_model]``."""

    def forward_layers(self, inputs_embeds: Tensor, attention_mask: Tensor, n_layers: int | None = None) -> Tensor:
        """Run ``n_layers`` layers over ``inputs_embeds`` (pre-norm embeddings).

        Returns the representation *after* the embedding norm, matching what
        ``inputs_embeds`` does on a full forward pass. No final norm is applied: the
        caller may want the pre-norm depth-n state, which is what hidden_states[n] is.
        """


class HuggingFaceSourceEncoder(SourceEncoder):
    """Works for ModernBERT and BERT-family encoders alike."""

    def __init__(self, model: nn.Module, model_id: str = "", layout: str = "memory") -> None:
        super().__init__()
        self.model = model
        # Recorded so a manifest built from this encoder can name what it byteified.
        # Empty when the caller loaded the weights some other way.
        self.model_id = model_id
        #: How the weights were found: ``checkpoint``, ``module`` or ``memory``. See
        #: :mod:`byteify.hub`; a model reached through a custom-code module layout is a
        #: different provenance from one ``AutoModel`` read at the repository root, and a
        #: build says which it used.
        self.layout = layout
        cfg = model.config
        self.d_model = int(cfg.hidden_size)
        self.vocab_size = int(cfg.vocab_size)

        if hasattr(model, "embeddings") and hasattr(model, "layers"):
            self._embeddings = model.embeddings
            self._layers = model.layers
            self._family = "modernbert"
        elif hasattr(model, "embeddings") and hasattr(model, "encoder") and hasattr(
            model.encoder, "layer"
        ):
            self._embeddings = model.embeddings
            self._layers = model.encoder.layer
            self._family = "bert"
        else:
            raise TypeError(
                f"unsupported encoder architecture {type(model).__name__}: expected "
                "`embeddings`+`layers` (ModernBERT) or `embeddings`+`encoder.layer` (BERT family)"
            )
        self.n_layers = len(self._layers)
        self._final_norm = getattr(model, "final_norm", None) or getattr(model, "norm", None)

    # -- introspection -------------------------------------------------------

    @property
    def family(self) -> str:
        return self._family

    @property
    def n_attention_heads(self) -> int:
        return int(getattr(self.model.config, "num_attention_heads", 8))

    @property
    def final_norm(self) -> nn.Module | None:
        return self._final_norm

    def embedding_matrix(self) -> Tensor:
        if self._family == "modernbert":
            return self._embeddings.tok_embeddings.weight
        return self._embeddings.word_embeddings.weight

    def token_embedding_matrix_for_layer_norm(self) -> Tensor:
        """Weight matrix for the embedding norm, matched by attribute probing."""
        norm = getattr(self._embeddings, "norm", None) or getattr(
            self._embeddings, "LayerNorm", None
        )
        return norm.weight if norm is not None else torch.empty(0)

    # -- forward -------------------------------------------------------------

    def forward_layers(
        self, inputs_embeds: Tensor, attention_mask: Tensor, n_layers: int | None = None
    ) -> Tensor:
        n = self.n_layers if n_layers is None else n_layers
        if self._family == "modernbert":
            return self._forward_modernbert(inputs_embeds, attention_mask, n)
        return self._forward_bert(inputs_embeds, attention_mask, n)

    def _forward_modernbert(self, inputs_embeds: Tensor, attention_mask: Tensor, n: int) -> Tensor:
        from transformers.masking_utils import (
            create_bidirectional_mask,
            create_bidirectional_sliding_window_mask,
        )

        model = self.model
        h = self._embeddings(inputs_embeds=inputs_embeds)

        B, N, _ = h.shape
        position_ids = torch.arange(N, device=h.device).unsqueeze(0).expand(B, N)
        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": h,
            "attention_mask": attention_mask,
        }
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        pos = {
            layer_type: model.rotary_emb(h, position_ids, layer_type)
            for layer_type in set(model.config.layer_types)
        }
        for layer in self._layers[:n]:
            h = layer(
                h,
                attention_mask=masks[layer.attention_type],
                position_embeddings=pos[layer.attention_type],
            )
        return h

    def _forward_bert(self, inputs_embeds: Tensor, attention_mask: Tensor, n: int) -> Tensor:
        h = self._embeddings(inputs_embeds=inputs_embeds)
        extended = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer in self._layers[:n]:
            out = layer(h, attention_mask=extended)
            h = out[0] if isinstance(out, (tuple, list)) else out
        return h

    def representation_at(self, input_ids: Tensor, attention_mask: Tensor, depth: int) -> Tensor:
        """Teacher representation at exactly ``depth``, computing nothing beyond it.

        Cheaper in memory than :meth:`full_representations`, which materialises every
        hidden state. On Apple Silicon the wasted activations of a deeper model are a
        substantial share of peak RSS, so training code should prefer this.
        """
        h = self._embeddings(input_ids=input_ids)
        if depth == 0:
            return h
        if self._family == "modernbert":
            from transformers.masking_utils import (
                create_bidirectional_mask,
                create_bidirectional_sliding_window_mask,
            )

            model = self.model
            B, N, _ = h.shape
            position_ids = torch.arange(N, device=h.device).unsqueeze(0).expand(B, N)
            mask_kwargs = {
                "config": model.config,
                "inputs_embeds": h,
                "attention_mask": attention_mask,
            }
            masks = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }
            pos = {
                lt: model.rotary_emb(h, position_ids, lt)
                for lt in set(model.config.layer_types)
            }
            for layer in self._layers[:depth]:
                h = layer(
                    h,
                    attention_mask=masks[layer.attention_type],
                    position_embeddings=pos[layer.attention_type],
                )
            return h

        extended = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer in self._layers[:depth]:
            out = layer(h, attention_mask=extended)
            h = out[0] if isinstance(out, (tuple, list)) else out
        return h

    def full_representations(self, input_ids: Tensor, attention_mask: Tensor) -> dict[int, Tensor]:
        """Teacher representations at depth 0..n_layers, for Stage 1 targets."""
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        states = {0: self._embeddings(input_ids=input_ids)}
        for i, h in enumerate(out.hidden_states, start=1):
            states[i] = h
        return states


class ByteifiedSourceEncoder(SourceEncoder):
    """A byteified model that has itself become the source for further work.

    Keeps ``SourceEncoder`` a real interface rather than a naming convention: once a
    byteified model is trained, the same Stage 1 recipe can be applied to it, and
    task arithmetic needs both models' transformer layers addressable the same way.
    """

    def __init__(self, encoder, source: SourceEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.source = source
        self.d_model = source.d_model
        self.n_layers = source.n_layers
        self.vocab_size = source.vocab_size

    def embedding_matrix(self) -> Tensor:
        return self.encoder.embed.suffix.weight

    def forward_layers(self, inputs_embeds, attention_mask, n_layers=None):
        return self.source.forward_layers(inputs_embeds, attention_mask, n_layers)


def resolve_source_encoder(model_or_name) -> HuggingFaceSourceEncoder:
    """Load a model by id (or wrap an already-loaded one) as a SourceEncoder.

    Accepts a clear error rather than a traceback for a model this repository cannot
    byteify, naming what is accepted. Both repository layouts resolve -- a plain
    checkpoint, and a custom-code module that keeps its encoder in a subdirectory -- see
    :mod:`byteify.hub`, which is where the second one is implemented.
    """
    from .hub import resolve_source_model

    if isinstance(model_or_name, str):
        model_id = model_or_name
        resolved = resolve_source_model(model_or_name)
        model = resolved.model
        layout = resolved.layout
    else:
        model = model_or_name
        model_id = getattr(model_or_name, "name_or_path", "") or ""
        layout = "memory"
    try:
        return HuggingFaceSourceEncoder(model, model_id=model_id, layout=layout)
    except TypeError as exc:
        raise TypeError(
            f"{model_id or type(model).__name__} is not a byteify source model. Accepted "
            "architectures: ModernBERT (`embeddings` + `layers`) and the BERT family "
            f"(`embeddings` + `encoder.layer`). Original error: {exc}"
        ) from exc


@dataclass
class ByteifyConfig:
    """Everything needed to byteify a given source model.

    Only ``model_id`` is usually supplied; dimensions are derived from the model so
    that pointing this at a different embedding model is a one-line change.
    """

    model_id: str
    # Local encoder. Bolmo keeps d_local == d_global because an up-projection
    # bottlenecked the representation rank; the suffix embeddings do most of the work,
    # so the local encoder can be small. Our sweeps found 0.66M and 3.67M parameters
    # within noise of each other.
    local_layers: int = 1
    local_backend: str = "attention"
    local_window: int = 128
    local_ffn_hidden: int = 1024
    # Boundary predictor. "mlp" is the full-context form that the encoder setting
    # makes possible; "cosine" reproduces Bolmo's 1-byte-lookahead predictor.
    boundary_predictor: str = "mlp"
    # Optional output-side local encoder (bidirectional). Bolmo's local decoder exists
    # to produce next-byte logits; for an encoder it exists to turn depooled patch
    # representations back into contextualised byte representations for Stage 2 MLM
    # and for byte-level pooling of the embedding.
    out_layers: int = 0
    out_ffn_hidden: int | None = None
    # Whether [CLS]/[SEP] style tokens are kept as pseudo-byte patches.
    add_special_tokens: bool = True
    # Freezing
    freeze_source_in_stage1: bool = True
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.model_id}: local={self.local_backend}x{self.local_layers}"
            f"(ffn={self.local_ffn_hidden}, win={self.local_window}), "
            f"boundary={self.boundary_predictor}, out_layers={self.out_layers}"
        )


def build_local_encoder(cfg: ByteifyConfig, source: SourceEncoder) -> nn.Module:
    from .model import LocalContextEncoder

    return LocalContextEncoder(
        source.d_model,
        n_layers=cfg.local_layers,
        ffn_hidden=cfg.local_ffn_hidden,
        n_heads=source.n_attention_heads if isinstance(source, HuggingFaceSourceEncoder) else 8,
        window=cfg.local_window,
        backend=cfg.local_backend,
    )


@dataclass(frozen=True)
class LocalEncoderSpec:
    """The byte interface's own architecture, as data rather than as a call-site constant.

    These are the settings a caller chooses; everything else about a byteified encoder is
    derived from the source model. They are recorded in a checkpoint and read back, so a
    checkpoint trained with anything other than the defaults can still be loaded: building
    the interface from defaults instead produced a different shape and failed the load.

    ``out_layers``/``out_ffn_hidden`` describe the *output* side -- the depooling path and
    the local decoder that returns contextualised representations to the byte axis. Stage 2
    needs it to predict masked bytes, and because it adds parameters it has to travel
    through the same spec: an encoder rebuilt without it returns ``byte_context=None``,
    which is the shape error surfacing somewhere far from its cause.
    """

    layers: int = 1
    backend: str = "attention"
    window: int = 128
    ffn_hidden: int = 1024
    boundary_predictor: str = "mlp"
    heads: int | None = None
    out_layers: int = 0
    out_ffn_hidden: int = 1024

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LocalEncoderSpec:
        """Read a spec from either a checkpoint's config or a manifest's ``encoder_arch``.

        Checkpoints store the flags under their ``local_*`` names; a manifest stores the
        spec's own field names. Reading both means a loader can be handed either without
        a translation step that could silently drop a field.
        """
        def pick(*names: str, default: Any) -> Any:
            for name in names:
                if name in config:
                    return config[name]
            return default

        return cls(
            layers=int(pick("layers", "local_layers", default=1)),
            backend=str(pick("backend", "local_backend", default="attention")),
            window=int(pick("window", "local_window", default=128)),
            ffn_hidden=int(pick("ffn_hidden", "local_ffn", default=1024)),
            boundary_predictor=str(pick("boundary_predictor", default="mlp")),
            out_layers=int(pick("out_layers", default=0)),
            out_ffn_hidden=int(pick("out_ffn_hidden", "out_ffn", default=1024)),
        )

    def as_dict(self) -> dict[str, Any]:
        """The spec under its own field names, for a manifest.

        A manifest is a contract read by something that does not know how a checkpoint
        happened to be written, so it carries the spec's names rather than the trainer's.
        """
        return {
            "layers": self.layers,
            "backend": self.backend,
            "window": self.window,
            "ffn_hidden": self.ffn_hidden,
            "boundary_predictor": self.boundary_predictor,
            "out_layers": self.out_layers,
            "out_ffn_hidden": self.out_ffn_hidden,
        }

    def to_config(self) -> dict[str, Any]:
        """The spec under the trainers' ``local_*`` names, for a checkpoint."""
        return {
            "local_layers": self.layers,
            "local_backend": self.backend,
            "local_window": self.window,
            "local_ffn": self.ffn_hidden,
            "boundary_predictor": self.boundary_predictor,
            "out_layers": self.out_layers,
            "out_ffn": self.out_ffn_hidden,
        }


def build_encoder_from_dimensions(
    *,
    d_model: int,
    n_byte_values: int,
    subword_vocab_size: int,
    spec: LocalEncoderSpec | None = None,
    n_heads: int = 8,
):
    """Build a byteified encoder from resolved numbers.

    The one place an encoder is constructed. Both routes into it -- a source model plus
    its tokenizer, and a manifest -- resolve their numbers and come through here, so the
    two cannot build different architectures out of the same facts.
    """
    from .model import ByteifiedEncoder, LocalContextEncoder

    spec = spec or LocalEncoderSpec()

    return ByteifiedEncoder(
        d_model=d_model,
        n_byte_values=n_byte_values,
        subword_vocab_size=subword_vocab_size,
        local_encoder=LocalContextEncoder(
            d_model,
            n_layers=spec.layers,
            ffn_hidden=spec.ffn_hidden,
            n_heads=n_heads,
            window=spec.window,
            backend=spec.backend,
        ),
        boundary_predictor=spec.boundary_predictor,
        local_decoder_layers=spec.out_layers,
        local_decoder_ffn_hidden=spec.out_ffn_hidden,
        local_decoder_window=spec.window,
        n_heads=n_heads,
        backend=spec.backend,
    )


def build_byteified_encoder(source: SourceEncoder, tokenizer, spec: LocalEncoderSpec | None = None):
    """Reconstruct a byteified encoder from a source model and its tokenizer.

    A convenience over :func:`build_encoder_from_dimensions`: it resolves the tokenizer's
    special tokens and vocabulary for you. Nothing about the interface is repeated here.
    """
    from .alignment import discover_special_bytes
    from .vocab import resolve_token_bytes

    spec = spec or LocalEncoderSpec()
    resolution = resolve_token_bytes(tokenizer)
    specials = discover_special_bytes(tokenizer)
    heads = spec.heads or getattr(source, "n_attention_heads", 8)

    return build_encoder_from_dimensions(
        d_model=source.d_model,
        n_byte_values=256 + len(specials),
        subword_vocab_size=resolution.vocab_size,
        spec=spec,
        n_heads=heads,
    )


def load_encoder_state(encoder: nn.Module, state: dict[str, Tensor], *, source: str = "checkpoint"):
    """Load encoder weights, refusing a silent partial load.

    ``strict=False`` is needed because a Stage 2 encoder can carry an output side a Stage 1
    encoder has no use for. But a shape disagreement is still an error: that is the point.
    A checkpoint holding weights this encoder cannot accept means the two disagree about
    their architecture, and loading the overlap would hide that.
    """
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            f"{source} holds weights the encoder has no use for: {sorted(unexpected)}"
        )
    return missing


def load_byteified_encoder(path, source: SourceEncoder, tokenizer):
    """Load a byteified encoder from a checkpoint, taking its architecture from the file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    recorded = checkpoint.get("config") or checkpoint.get("args") or {}
    encoder = build_byteified_encoder(source, tokenizer, LocalEncoderSpec.from_config(recorded))
    load_encoder_state(encoder, checkpoint["encoder"], source=str(path))
    return encoder