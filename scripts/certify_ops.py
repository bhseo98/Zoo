#!/usr/bin/env python3
"""Certify aten ops for the on-device allowlist — with a compile, not a guess.

The audit allowlist (`analysis/op_audit.py`) claims an op "lowers cleanly". This
script is the evidence behind that claim: one tiny probe module per op family,
each exported to torch-dialect MLIR **and compiled to a CPU vmfb**. Ops emitted
by a probe that compiled are certified; anything else stays unknown.

Tiny probes on purpose — a 1B checkpoint proves nothing an 8x8 tensor does not,
and these run in seconds so the certification is repeatable.

Run (inside venv-shark):
    python scripts/certify_ops.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from torch_mlir_zoo import export_for_npu  # noqa: E402
from torch_mlir_zoo.analysis import SUPPORTED_ATEN_OPS  # noqa: E402

OUT = ROOT / "logs" / "op-certification.json"


class LayerNormProbe(nn.Module):
    """var_mean / rsqrt / sub — the LayerNorm family (gpt2, bert, opt, whisper)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(x)


class Conv1dProbe(nn.Module):
    """convolution — whisper's audio front end."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class MaskProbe(nn.Module):
    """gt / eq / where / fill — causal-mask construction byproducts."""

    def forward(self, x):
        mask = x > 0
        zeroed = torch.where(mask, x, torch.zeros_like(x))
        return zeroed + (x == 0).to(x.dtype)


class SliceScatterProbe(nn.Module):
    """slice_scatter / copy — in-place window writes (opt's mask path)."""

    def forward(self, x):
        y = x.clone()
        y[:, :2] = 0.0
        return y


class CumsumProbe(nn.Module):
    """cumsum — opt's position ids from the attention mask."""

    def forward(self, x):
        return torch.cumsum(x, dim=-1)


class RepeatProbe(nn.Module):
    """repeat / expand — head/beam broadcasting."""

    def forward(self, x):
        return x.repeat(1, 2)


class RsubProbe(nn.Module):
    """rsub — bert's (1.0 - mask) attention bias."""

    def forward(self, x):
        return 1.0 - x


class InplaceCopyProbe(nn.Module):
    """copy / empty_strided — buffer allocation + in-place write byproducts."""

    def forward(self, x):
        buf = torch.empty_like(x)
        buf.copy_(x)
        return buf * 2


class DtypeCastProbe(nn.Module):
    """_to_copy — the functional form of `.to(dtype)`.

    Certified under the fused profile on purpose: with decompositions on it
    collapses into `to`, so the analysis profile never shows it.
    """

    def forward(self, x):
        return x.to(torch.float32) * 2.0


class DropoutProbe(nn.Module):
    """dropout — identity in eval, but only the fused profile still shows it.

    Every HF checkpoint carries it, so leaving it uncertified made the fused
    audit red on all of them for an op that costs nothing to execute.
    """

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(x)


# probe -> (factory, input shape, export profile)
PROBES = {
    "dtype_cast": (DtypeCastProbe, (2, 8), "fused"),
    "dropout": (DropoutProbe, (2, 8), "fused"),
    "layernorm": (LayerNormProbe, (2, 4, 16)),
    # the whole-op forms, which only survive when decompositions are skipped
    "layernorm_fused": (LayerNormProbe, (2, 4, 16), "fused"),
    "inplace_copy": (InplaceCopyProbe, (2, 8)),
    "conv1d": (Conv1dProbe, (1, 4, 16)),
    "conv1d_fused": (Conv1dProbe, (1, 4, 16), "fused"),
    "mask": (MaskProbe, (2, 8)),
    "slice_scatter": (SliceScatterProbe, (2, 8)),
    "cumsum": (CumsumProbe, (2, 8)),
    "repeat": (RepeatProbe, (2, 8)),
    "rsub": (RsubProbe, (2, 8)),
}


def compile_to_vmfb(model: nn.Module, args: tuple) -> int:
    """Compile ``model`` to a host CPU vmfb; return its size in bytes."""
    from iree.turbine.aot import FxProgramsBuilder, export

    fxb = FxProgramsBuilder(model.eval())

    @fxb.export_program(name="forward", args=args, strict=False)
    def _entry(m, *runtime_args):  # noqa: ANN001
        return m(*runtime_args)

    output = export(fxb)
    vmfb = output.compile(save_to=None, target_backends=("llvm-cpu",))
    return len(bytes(vmfb.map_memory()))


def main() -> int:
    torch.manual_seed(0)
    rows = []
    certified: set[str] = set()

    for name, spec in PROBES.items():
        factory, shape, *rest = spec
        profile = rest[0] if rest else "analysis"
        model, args = factory(), (torch.randn(*shape),)
        row: dict = {"probe": name, "profile": profile}
        try:
            r = export_for_npu(model, args, backend="iree_turbine", profile=profile)
            row["ops"] = sorted(r.summary["op_counts"])
            row["vmfb_bytes"] = compile_to_vmfb(model, args)
            row["status"] = "certified"
            certified.update(row["ops"])
        except Exception as e:
            row["status"] = "fail"
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rows.append(row)
        print(f"{name:15} {row['status']:10} {row.get('ops', row.get('error', ''))}")

    new_ops = sorted(certified - set(SUPPORTED_ATEN_OPS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"probes": rows, "newly_certified": new_ops}, indent=2))

    print(f"\ncertified ops: {len(certified)}")
    print(f"not yet on the allowlist: {new_ops}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
