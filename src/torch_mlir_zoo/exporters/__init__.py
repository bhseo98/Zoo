"""Toolchain-isolated export entry points.

Two backends, both `(module, args) -> str`:
  * `export_top_level_torch_dialect` — `torch_mlir.compile(..., OutputType.TORCH)`
  * `export_via_iree_turbine` — `iree.turbine.aot.FxProgramsBuilder` + `aot.export`
The rest of the zoo (stages, analyzers, tests) sees a backend swap only.
"""
from .iree_turbine_export import export_via_iree_turbine
from .torch_mlir_export import export_top_level_torch_dialect

__all__ = ["export_top_level_torch_dialect", "export_via_iree_turbine"]
