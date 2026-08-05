"""L1 capture layer — get an arbitrary PyTorch model into a traceable shape, and
name the failure in the model's terms when it still will not trace.

Export failures sit on three layers, and only the last one is a kernel problem:

    L0 load     — weights never materialized (meta device / tied-weight bugs)
    L1 capture  — ``torch.export`` cannot build a graph (tracer limits, HF
                  wrapper objects, dynamic control flow)
    L2 op->IR   — a captured op has no torch-dialect lowering, or collapses into
                  one opaque composite (SDPA / flash) that hides the math

:func:`prepare` applies the rewrites we have *measured* to matter on L0-L2;
:func:`diagnose` maps a raised exception onto its layer plus a concrete next
step. A CustomOp is on none of these layers — that only enters when an op has no
standard representation at all (see ``kernels/block_scaled_q8.py``).
"""
from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "Diagnosis",
    "ExportError",
    "ForwardOnly",
    "diagnose",
    "drop_context_markers",
    "prepare",
]

_META_HINT = (
    "weights are on the meta device — a load-time bug, not an export one. For a "
    "checkpoint that stores only the tied copy (e.g. facebook/opt-125m ships "
    "lm_head.weight and no embed_tokens.weight), reload with "
    "tie_word_embeddings=False and re-tie by hand: "
    "model.get_input_embeddings().weight = model.get_output_embeddings().weight"
)


@dataclass(frozen=True)
class Diagnosis:
    """Which layer failed, why, and what to do about it."""

    layer: str
    cause: str
    hint: str
    retry: str | None = None  # capture strategy worth retrying, if any

    def __str__(self) -> str:
        return f"[{self.layer}] {self.cause}\n  fix: {self.hint}"


class ExportError(RuntimeError):
    """An export that failed, carrying its :class:`Diagnosis`."""

    def __init__(self, diagnosis: Diagnosis, original: BaseException | None = None):
        super().__init__(str(diagnosis))
        self.diagnosis = diagnosis
        self.original = original


# --------------------------------------------------------------------------
# Diagnosis — error signature -> layer / cause / fix
# --------------------------------------------------------------------------

# (substrings that must all appear, layer, cause, hint, retry-strategy)
_RULES: tuple[tuple[tuple[str, ...], str, str, str, str | None], ...] = (
    (
        ("AttrProxy", "missing 1 required positional"),
        "L1-capture",
        "nn.ModuleList sliced inside forward (e.g. self.layers[:n]) — the "
        "non-strict export tracer cannot rebuild the proxied container",
        "retry with the strict capture strategy (export_for_npu does this "
        "automatically); measured to fix Qwen2.5 / Llama-family HF models",
        "strict",
    ),
    (
        ("Unhandled FakeTensor Device Propagation",),
        "L0-load",
        "an operand is on the meta device while the rest are on cpu",
        _META_HINT,
        None,
    ),
    (
        ("scaled_dot_product",),
        "L2-lowering",
        "attention collapsed into one opaque SDPA/flash composite op",
        "force eager attention so it decomposes into bmm + softmax "
        "(export_for_npu does this when rewrite=True)",
        None,
    ),
    (
        ("data-dependent",),
        "L1-capture",
        "a shape depends on tensor *values*, so the graph is not static",
        "give the model fixed example_args and remove value-dependent slicing "
        "(on-device targets need static shapes anyway)",
        None,
    ),
    (
        ("Dynamic control flow",),
        "L1-capture",
        "python control flow branches on a traced tensor",
        "rewrite the branch out of forward, or wrap that region in a CustomOp "
        "so the tracer sees one opaque call",
        None,
    ),
)


def diagnose(exc: BaseException) -> Diagnosis:
    """Classify ``exc`` into the layer that actually failed."""
    text = f"{type(exc).__name__}: {exc}"
    for needles, layer, cause, hint, retry in _RULES:
        if all(n in text for n in needles):
            return Diagnosis(layer, cause, hint, retry)
    return Diagnosis(
        "unclassified",
        f"{type(exc).__name__}: {str(exc)[:200]}",
        "not a known signature — run the export directly for the full traceback, "
        "then add the signature to torch_mlir_zoo.capture._RULES",
    )


# --------------------------------------------------------------------------
# Rewrites
# --------------------------------------------------------------------------


def meta_params(model: nn.Module) -> list[str]:
    """Names of parameters/buffers still on the meta device (never loaded)."""
    named = list(model.named_parameters(remove_duplicate=False))
    named += list(model.named_buffers(remove_duplicate=False))
    return [n for n, t in named if t.device.type == "meta"]


def _first_tensor(out: Any) -> torch.Tensor:
    """Pull the primary tensor out of a HF ModelOutput / tuple / tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return _first_tensor(out[0])
    for name in ("logits", "last_hidden_state", "prediction_logits"):
        value = getattr(out, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"cannot reduce {type(out).__name__} to a single output tensor")


class ForwardOnly(nn.Module):
    """Wrap a model so ``forward(*args)`` returns one tensor.

    ``use_cache=False`` drops the KV-cache branch — the source of the cache /
    paged ops the on-device contract rejects — and ``return_dict=False`` drops
    the ModelOutput wrapper the tracer cannot return.

    Args:
        model: the module to wrap.
        arg_names: bind positional args to these keyword names instead. Needed
            when the interesting inputs are not the leading positionals — e.g.
            Whisper takes ``(input_features, attention_mask, decoder_input_ids)``,
            so ``("input_features", "decoder_input_ids")`` skips the mask.
    """

    def __init__(self, model: nn.Module, arg_names: tuple[str, ...] | None = None):
        super().__init__()
        self.model = model
        self.arg_names = arg_names
        params = inspect.signature(model.forward).parameters
        self.call_kwargs = {k: False for k in ("use_cache", "return_dict") if k in params}

    def forward(self, *args):
        if self.arg_names is None:
            return _first_tensor(self.model(*args, **self.call_kwargs))
        bound = dict(zip(self.arg_names, args, strict=True))
        return _first_tensor(self.model(**bound, **self.call_kwargs))


def _force_eager_attention(model: nn.Module) -> bool:
    """Set every (sub)config to eager attention. True when something changed.

    SDPA / flash implementations export as one opaque composite op
    (``_scaled_dot_product_flash_attention_for_cpu``); eager decomposes into
    bmm + softmax, which every backend can lower.
    """
    configs = []
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        configs.append(config)
        for name in ("text_config", "vision_config", "audio_config", "encoder", "decoder"):
            sub = getattr(config, name, None)
            if hasattr(sub, "_attn_implementation"):
                configs.append(sub)

    changed = False
    for config in configs:
        if getattr(config, "_attn_implementation", "eager") != "eager":
            try:
                config._attn_implementation = "eager"
            except (AttributeError, ValueError):  # config rejects the assignment
                continue
            changed = True
    return changed


def _is_hf_style(model: nn.Module) -> bool:
    """True when forward takes the HF cache/dict knobs — the marker we rewrite for."""
    params = inspect.signature(model.forward).parameters
    return "use_cache" in params or "return_dict" in params


def prepare(
    model: nn.Module,
    arg_names: tuple[str, ...] | None = None,
    keep_fused: bool = False,
) -> tuple[nn.Module, list[str]]:
    """Apply the known capture rewrites; return ``(model, applied_names)``.

    Args:
        model: the module to make traceable.
        arg_names: keyword names to bind the example args to, for models whose
            leading positionals are not the inputs (Whisper — see
            :class:`ForwardOnly`). Positional binding otherwise.
        keep_fused: skip the eager-attention rewrite. Decomposing SDPA is right
            for analysis but wrong for the fused-op contract, whose pattern matcher
            wants one ``aten.scaled_dot_product_attention``.

    Raises:
        ExportError: when the model cannot be fixed by a rewrite — currently
            only unmaterialized (meta) weights, which no export can recover.
    """
    unloaded = meta_params(model)
    if unloaded:
        raise ExportError(
            Diagnosis(
                "L0-load",
                f"{len(unloaded)} tensor(s) never materialized: {unloaded[:4]}",
                _META_HINT,
            )
        )

    applied: list[str] = []
    if model.training:
        model = model.eval()
        applied.append("eval_mode")
    if not keep_fused and _force_eager_attention(model):
        applied.append("eager_attention")
    if not isinstance(model, ForwardOnly) and _is_hf_style(model):
        model = ForwardOnly(model, arg_names=arg_names)
        applied.append("forward_only")
    return model, applied


# --------------------------------------------------------------------------
# Context markers
# --------------------------------------------------------------------------

# torch.export records a `with torch.autocast(...)` / `with torch.no_grad()`
# region as marker nodes, then wraps the region in a higher-order op so the
# ExportedProgram can re-establish the context if it is ever run eagerly. We
# never run it — we import it — and the fx_importer has no lowering for a HOP:
#
#     NotImplementedError: Higher-order operation 'wrap_with_autocast'
#
# ``run_decompositions`` dissolves the HOPs, which is why the analysis profile
# never hit this; the fused profile skips decomposition to keep linear /
# rms_norm / sdpa whole, so the HOP survives to the importer and every
# llama-family checkpoint stops there. Measured markers: the HF rotary embedding
# opens ``autocast(enabled=False)``, and every HF forward opens ``no_grad()``.
_ENTER_AUTOCAST = torch.amp.autocast_mode._enter_autocast
_EXIT_AUTOCAST = torch.amp.autocast_mode._exit_autocast
_SET_GRAD_ENABLED = torch._C._set_grad_enabled


def _droppable_markers(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """The marker nodes in ``graph`` that carry no math, so erasing is exact.

    Grad mode never changes a forward value. Autocast does — but only when it
    is *on*, and the region we meet in practice turns it off. An enabled block
    is left alone so it fails loudly rather than silently changing dtypes.
    """
    drop: set[torch.fx.Node] = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is _SET_GRAD_ENABLED:
            drop.add(node)
        # _enter_autocast(device_type, dtype, enabled, cache_enabled) — the
        # slice reads "enabled is False", and a missing arg means the default,
        # which is True.
        elif node.target is _ENTER_AUTOCAST and node.args[2:3] == (False,):
            drop.add(node)
        elif node.target is _EXIT_AUTOCAST and node.args[0] in drop:
            drop.add(node)
    return drop


def _drop_markers_first(original):
    """Wrap one of torch's ``replace_*_with_hop_pass`` — erase, then delegate.

    Same signature as the pass it wraps. Markers we cannot erase are left for
    ``original``, which still turns them into a HOP: a marker the verifier
    finds unconsumed is a hard error, so skipping one is not an option. Nothing
    else about the graph moves, so the fused aten ops the fused profile exists
    for are untouched.
    """

    def _pass(gm: torch.fx.GraphModule, graph_signature):
        for mod in gm.modules():
            if not isinstance(mod, torch.fx.GraphModule):
                continue
            drop = _droppable_markers(mod.graph)
            if not drop:
                continue
            # Reverse order: an _exit_autocast is the only user of its _enter.
            for node in reversed(list(mod.graph.nodes)):
                if node in drop:
                    mod.graph.erase_node(node)
            mod.recompile()
        gm.graph.lint()
        return original(gm, graph_signature)

    return _pass


@contextlib.contextmanager
def drop_context_markers():
    """Export inside this to get a graph with no autocast / no_grad HOPs.

    ``torch.export._trace`` imports the two passes at call time, so replacing
    the module attribute is what reaches them. Monkeypatching the context
    managers themselves does not: the HOP is built by an export pass, not by
    ``__enter__``.
    """
    import torch._export.passes.replace_autocast_with_hop_pass as autocast_pass
    import torch._export.passes.replace_set_grad_with_hop_pass as set_grad_pass

    saved = (
        autocast_pass.replace_autocast_with_hop_pass,
        set_grad_pass.replace_set_grad_with_hop_pass,
    )
    autocast_pass.replace_autocast_with_hop_pass = _drop_markers_first(saved[0])
    set_grad_pass.replace_set_grad_with_hop_pass = _drop_markers_first(saved[1])
    try:
        yield
    finally:
        (
            autocast_pass.replace_autocast_with_hop_pass,
            set_grad_pass.replace_set_grad_with_hop_pass,
        ) = saved
