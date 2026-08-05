"""Alpaca CLI — the ``alpaca`` command.

A thin Click wrapper over the SDK facade (``export_for_npu`` / ``summarize``).
Registered as a console entry point in ``pyproject.toml``:

    [project.scripts]
    alpaca = "torch_mlir_zoo.cli:main"

Subcommands:
    alpaca export <op> [--backend ...] [--quantize int8] [--verify] [-o out.mlir]
    alpaca estimate <model.mlir>          # summarize an exported IR (contract gate)
    alpaca models                         # list zoo ops the CLI can export
"""
from __future__ import annotations

import json
import sys

import click

from . import __version__

# name -> (factory dotted-path, example-args builder). Kept in sync with the GUI's MODELS.
_ZOO = {
    "attention": ("ScaledDotProductAttention", (1, 32, 512)),
    "rmsnorm": ("RMSNorm", (1, 32, 512)),
    "mlp": ("SwiGLU", (1, 32, 512)),
    "topk": ("TopK", (1, 32000)),
}


def _build(op: str):
    import torch

    from . import ops

    factory_name, shape = _ZOO[op]
    factory = getattr(ops, factory_name)
    return factory(), (torch.randn(*shape),)


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"], max_content_width=100)
)
@click.version_option(version=__version__, message="Alpaca SDK version: %(version)s")
def main() -> None:
    """Alpaca — on-device compiler frontend (PyTorch -> torch-dialect MLIR)."""


@main.command("models")
def models_cmd() -> None:
    """List the zoo ops this CLI can export."""
    for name, (factory, shape) in _ZOO.items():
        click.echo(f"  {name:10} {factory:28} example {shape}")


@main.command("export")
@click.argument("op", type=click.Choice(list(_ZOO)))
@click.option(
    "--backend",
    type=click.Choice(["torch_mlir", "iree_turbine"]),
    default="iree_turbine",
    show_default=True,
    help="torch_mlir is kept for completeness but its wheel index is dead.",
)
@click.option("--quantize", type=click.Choice(["int8"]), default=None, help="Apply INT8 block_scaled_q8.")
@click.option("--verify", is_flag=True, help="With --quantize int8, report fp-vs-int8 accuracy.")
@click.option("--no-rewrite", is_flag=True, help="Export the model as-is (skip capture rewrites).")
@click.option(
    "--profile",
    type=click.Choice(["analysis", "fused"]),
    default="analysis",
    show_default=True,
    help="Decomposed IR for the audit, or fused IR for the target NPU pattern matcher.",
)
@click.option("-o", "--out", type=click.Path(dir_okay=False), help="Write MLIR to this path.")
def export_cmd(
    op: str,
    backend: str,
    quantize: str | None,
    verify: bool,
    no_rewrite: bool,
    profile: str,
    out: str | None,
) -> None:
    """Compile a zoo OP to top-level torch-dialect MLIR."""
    from .alpaca import export_for_npu
    from .capture import ExportError

    model, args = _build(op)
    try:
        r = export_for_npu(
            model, args, backend=backend, quantize=quantize, verify=verify,
            rewrite=not no_rewrite, profile=profile,
        )
    except ModuleNotFoundError as e:
        raise click.ClickException(f"backend {backend!r} toolchain not installed: {e}")
    except ExportError as e:
        raise click.ClickException(f"export failed\n  {e.diagnosis}")

    badge = "OK" if r.ok else "SERVER-SIDE OPS"
    click.secho(f"[{badge}] {op} -> {backend} ({r.summary['n_lines']} lines)", fg="green" if r.ok else "red")
    click.echo(f"  server_side_op_hits = {r.summary['server_side_op_hits']}")
    if r.rewrites or r.capture:
        click.echo(
            f"  profile={r.profile}  capture={r.capture or 'n/a'}  "
            f"rewrites={r.rewrites or '[]'}"
        )
    if r.accuracy is not None:
        a = r.accuracy
        click.echo(f"  accuracy: max_rel={a['max_rel']:.4f}  cosine={a['cosine']:.5f}")
    if out:
        r.save(out)
        click.echo(f"  written: {out}")
    sys.exit(0 if r.ok else 2)


@main.command("estimate")
@click.argument("mlir", type=click.File("r"))
def estimate_cmd(mlir) -> None:
    """Summarize an exported .mlir — op counts + contract gate (exit 2 if server-side)."""
    from .analysis import summarize

    s = summarize(mlir.read())
    click.echo(json.dumps(s, indent=2, ensure_ascii=False))
    ok = s.get("server_side_op_hits", {}) == {}
    click.secho("on-device: OK" if ok else "on-device: FAIL (server-side ops)", fg="green" if ok else "red")
    sys.exit(0 if ok else 2)


@main.command("audit")
@click.argument("mlir", type=click.File("r"))
@click.option(
    "--profile",
    type=click.Choice(["analysis", "fused"]),
    default="analysis",
    show_default=True,
    help="Profile the IR was exported with — fused profiles keep whole ops.",
)
def audit_cmd(mlir, profile: str) -> None:
    """Audit an exported .mlir against the on-device op allowlist.

    Exit 2 when a server-side op leaks or an unlisted (unknown) op appears.
    """
    from .analysis import audit as run_audit

    rep = run_audit(mlir.read(), profile=profile)
    click.echo(f"  unique ops: {rep['unique_ops']}  supported: {len(rep['supported'])}")
    if rep["unknown"]:
        click.secho(f"  unknown ops: {rep['unknown']}", fg="yellow")
    if rep["server_side"]:
        click.secho(f"  server-side: {rep['server_side']}", fg="red")
    click.secho(
        "on-device: OK" if rep["ok"] else "on-device: FAIL",
        fg="green" if rep["ok"] else "red",
    )
    sys.exit(0 if rep["ok"] else 2)


if __name__ == "__main__":
    main()
