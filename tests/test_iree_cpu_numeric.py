"""Self-contained numeric verification: zoo models export → IREE (llvm-cpu) →
run → match PyTorch. Proves the lowering is correct through the *standard* IREE
path (no target-specific custom kernel), entirely on the host CPU.

Skipped unless torch + iree.turbine + iree.compiler/runtime are installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("iree.turbine")
ireec = pytest.importorskip("iree.compiler")
ireert = pytest.importorskip("iree.runtime")

import torch.nn as nn  # noqa: E402

from torch_mlir_zoo.ops import RMSNorm, SwiGLU, ScaledDotProductAttention  # noqa: E402
from torch_mlir_zoo.exporters.iree_turbine_export import export_via_iree_turbine  # noqa: E402


def _init(m):
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(mean=0.0, std=0.05)
    return m


class _DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block assembled purely from zoo ops."""

    def __init__(self, dim=256, heads=8, kv=2, hidden=512):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = ScaledDotProductAttention(dim, heads, kv)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class _TopK(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.topk(x, self.k, dim=-1)


def _run_iree_cpu(mlir: str, np_inputs):
    vmfb = ireec.compile_str(mlir, target_backends=["llvm-cpu"])
    cfg = ireert.Config("local-task")
    vm = ireert.VmModule.copy_buffer(cfg.vm_instance, vmfb)
    ctx = ireert.SystemContext(config=cfg)
    ctx.add_vm_module(vm)
    res = getattr(ctx.modules, vm.name).forward(*np_inputs)
    return list(res) if isinstance(res, (list, tuple)) else [res]


CASES = [
    ("rmsnorm", lambda: _init(RMSNorm(512)), lambda: (torch.randn(1, 8, 512),)),
    ("swiglu", lambda: _init(SwiGLU(512, 1024)), lambda: (torch.randn(1, 8, 512),)),
    (
        "attention_gqa",
        lambda: _init(ScaledDotProductAttention(512, 8, 2)),
        lambda: (torch.randn(1, 8, 512),),
    ),
    ("topk", lambda: _TopK(5), lambda: (torch.randn(1, 64),)),
    ("decoder_block", lambda: _init(_DecoderBlock()), lambda: (torch.randn(1, 16, 256),)),
]


@pytest.mark.parametrize("name,make,make_args", CASES, ids=[c[0] for c in CASES])
def test_iree_cpu_matches_torch(name, make, make_args):
    torch.manual_seed(0)
    model = make().eval()
    args = make_args()

    ref = model(*args)
    ref = [r.detach().numpy() for r in (ref if isinstance(ref, (tuple, list)) else [ref])]

    mlir = export_via_iree_turbine(model, args, func_name="forward")
    outs = [np.asarray(o) for o in _run_iree_cpu(mlir, [a.numpy() for a in args])]

    assert len(outs) == len(ref)
    for r, o in zip(ref, outs):
        if np.issubdtype(r.dtype, np.integer) or np.issubdtype(o.dtype, np.integer):
            assert np.array_equal(r.astype(np.int64), o.astype(np.int64))
        else:
            assert np.allclose(r, o, atol=1e-4, rtol=1e-4), (
                f"{name}: max_abs_err={np.max(np.abs(r - o)):.3e}"
            )
