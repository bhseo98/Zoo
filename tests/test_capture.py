"""L1 capture layer — rewrites applied before tracing + failure diagnosis."""
import pytest
import torch
import torch.nn as nn

from torch_mlir_zoo.capture import (
    Diagnosis,
    ExportError,
    ForwardOnly,
    diagnose,
    drop_context_markers,
    meta_params,
    prepare,
)


class _FakeConfig:
    def __init__(self):
        self._attn_implementation = "sdpa"


class _FakeHF(nn.Module):
    """HF-shaped model: config with an attn impl, cache/dict knobs, tuple output."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.lin = nn.Linear(4, 4)

    def forward(self, x, use_cache=True, return_dict=True):
        y = self.lin(x)
        return (y, "kv-cache-state") if not return_dict else {"logits": y}


# --------------------------------------------------------------- diagnosis


def test_diagnose_modulelist_slicing_suggests_strict_retry():
    # The measured Qwen2.5-0.5B failure — self.layers[:n] under the non-strict tracer.
    exc = TypeError(
        "_ModuleStackTracer.__init__.<locals>.AttrProxy.__init__() "
        "missing 1 required positional argument: 'path'"
    )
    d = diagnose(exc)
    assert d.layer == "L1-capture"
    assert d.retry == "strict"


def test_diagnose_meta_device_is_a_load_problem():
    # The measured facebook/opt-125m failure — export blamed, loader at fault.
    exc = RuntimeError(
        "Unhandled FakeTensor Device Propagation for aten.native_layer_norm.default, "
        "found two different devices meta, cpu"
    )
    d = diagnose(exc)
    assert d.layer == "L0-load"
    assert d.retry is None
    assert "tie_word_embeddings" in d.hint


def test_diagnose_unknown_error_is_marked_unclassified():
    d = diagnose(RuntimeError("something nobody has seen"))
    assert d.layer == "unclassified"
    assert "_RULES" in d.hint  # tells the reader where to add the signature


def test_diagnosis_str_shows_layer_and_fix():
    text = str(Diagnosis("L1-capture", "cause text", "fix text"))
    assert "L1-capture" in text and "cause text" in text and "fix text" in text


# ---------------------------------------------------------------- rewrites


def test_meta_params_are_detected_and_block_export():
    m = nn.Linear(4, 4)
    m.weight = nn.Parameter(torch.empty(4, 4, device="meta"))
    assert meta_params(m) == ["weight"]
    with pytest.raises(ExportError) as e:
        prepare(m)
    assert e.value.diagnosis.layer == "L0-load"


def test_prepare_rewrites_hf_style_model():
    model, applied = prepare(_FakeHF().train())
    assert set(applied) == {"eval_mode", "eager_attention", "forward_only"}
    assert isinstance(model, ForwardOnly)
    assert model.model.config._attn_implementation == "eager"
    # forward now returns a bare tensor (cache state and dict wrapper gone)
    out = model(torch.randn(2, 4))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 4)


def test_prepare_leaves_a_plain_module_alone():
    # A zoo op is already capture-ready — nothing to rewrite.
    model, applied = prepare(nn.Linear(4, 4).eval())
    assert applied == []
    assert not isinstance(model, ForwardOnly)


def test_forward_only_unwraps_model_output_objects():
    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _M(nn.Module):
        def forward(self, x, return_dict=True):
            return _Out(x * 2)

    wrapped = ForwardOnly(_M())
    assert wrapped.call_kwargs == {"return_dict": False}
    assert torch.equal(wrapped(torch.ones(3)), torch.full((3,), 2.0))


# ------------------------------------------------------- context managers


class _Autocast(nn.Module):
    """A `with torch.autocast(...)` block between two Linears — the HF rotary
    shape. ``enabled`` picks whether the block turns autocast off or on."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        h = self.a(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=self.enabled):
            h = h * 3.0
        return self.b(h.float())


class _NoGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        y = self.lin(x)
        with torch.no_grad():
            return y * 2.0


def _hops(ep) -> list[str]:
    return [
        str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_function" and isinstance(n.target, torch._ops.HigherOrderOperator)
    ]


def test_autocast_off_block_exports_as_a_higher_order_op_without_the_fix():
    # The measured blocker: torch.export records the `with` block as a HOP, and
    # the fx_importer has no lowering for it.
    ep = torch.export.export(_Autocast(enabled=False).eval(), (torch.randn(2, 8),), strict=True)
    assert _hops(ep) == ["wrap_with_autocast"]


def test_drop_context_markers_removes_the_autocast_hop_and_keeps_the_values():
    model, args = _Autocast(enabled=False).eval(), (torch.randn(2, 8),)
    with drop_context_markers():
        ep = torch.export.export(model, args, strict=True)
    assert _hops(ep) == []
    assert torch.equal(ep.module()(*args), model(*args))


def test_drop_context_markers_removes_the_no_grad_hop_and_keeps_the_values():
    model, args = _NoGrad().eval(), (torch.randn(2, 8),)
    assert _hops(torch.export.export(model, args, strict=False)) == [
        "wrap_with_set_grad_enabled"
    ]
    with drop_context_markers():
        ep = torch.export.export(model, args, strict=False)
    assert _hops(ep) == []
    assert torch.equal(ep.module()(*args), model(*args))


def test_an_enabled_autocast_block_is_left_alone():
    # Dropping this one would silently change the numerics of everything inside
    # it, so it stays a HOP and the export fails loudly instead.
    with drop_context_markers():
        ep = torch.export.export(_Autocast(enabled=True).eval(), (torch.randn(2, 8),), strict=True)
    assert _hops(ep) == ["wrap_with_autocast"]


def test_drop_context_markers_restores_the_torch_passes():
    import torch._export.passes.replace_autocast_with_hop_pass as ac

    before = ac.replace_autocast_with_hop_pass
    with drop_context_markers():
        assert ac.replace_autocast_with_hop_pass is not before
    assert ac.replace_autocast_with_hop_pass is before


# ------------------------------------------------------------ SDK reporting


def test_export_reports_rewrites_and_capture_strategy():
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo import export_for_npu

    r = export_for_npu(_FakeHF(), (torch.randn(1, 4),), backend="iree_turbine")
    assert r.capture == "nonstrict"
    assert "forward_only" in r.rewrites and "eager_attention" in r.rewrites
    assert r.ok is True


def test_rewrite_false_keeps_the_model_as_handed_in():
    pytest.importorskip("iree.turbine")
    from torch_mlir_zoo import export_for_npu

    m = _FakeHF()
    r = export_for_npu(m, (torch.randn(1, 4),), backend="iree_turbine", rewrite=False)
    assert r.rewrites == []
    assert m.config._attn_implementation == "sdpa"  # config untouched
