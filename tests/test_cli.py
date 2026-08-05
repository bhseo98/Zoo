"""Alpaca CLI smoke tests (Click CliRunner)."""
import pytest
from click.testing import CliRunner

from torch_mlir_zoo.cli import main


def test_version():
    r = CliRunner().invoke(main, ["--version"])
    assert r.exit_code == 0
    assert "Alpaca SDK version" in r.output


def test_models_lists_zoo_ops():
    r = CliRunner().invoke(main, ["models"])
    assert r.exit_code == 0
    for op in ("attention", "rmsnorm", "mlp", "topk"):
        assert op in r.output


def test_export_bad_op_rejected():
    r = CliRunner().invoke(main, ["export", "nope"])
    assert r.exit_code != 0  # Click Choice rejects


def test_export_rmsnorm_turbine(tmp_path):
    pytest.importorskip("iree.turbine")
    out = tmp_path / "r.mlir"
    r = CliRunner().invoke(main, ["export", "rmsnorm", "--backend", "iree_turbine", "-o", str(out)])
    assert r.exit_code == 0
    assert "OK" in r.output
    assert out.exists() and "module" in out.read_text()


def test_estimate_clean_ir(tmp_path):
    pytest.importorskip("iree.turbine")
    out = tmp_path / "r.mlir"
    CliRunner().invoke(main, ["export", "rmsnorm", "--backend", "iree_turbine", "-o", str(out)])
    r = CliRunner().invoke(main, ["estimate", str(out)])
    assert r.exit_code == 0
    assert "on-device: OK" in r.output
