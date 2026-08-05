"""Tiny registry — `@register("<stage>", "<name>")` + `build("<stage>", config)`.

Two-level dict: stage -> name -> class. `build` consumes the config dict's
`type` key to pick the class, and forwards the rest as kwargs. Stage names
are arbitrary strings — the framework core has no built-in knowledge of any
particular domain (voice / vision / multimodal / ...).
"""
from __future__ import annotations

from typing import Any, Type

_REGISTRY: dict[str, dict[str, Type[Any]]] = {}


def register(stage: str, name: str):
    def decorator(cls: Type[Any]) -> Type[Any]:
        _REGISTRY.setdefault(stage, {})[name] = cls
        return cls

    return decorator


def build(stage: str, config: dict[str, Any]) -> Any:
    if stage not in _REGISTRY:
        raise KeyError(f"Unknown stage '{stage}'. Known: {list(_REGISTRY)}")
    cfg = dict(config)
    type_name = cfg.pop("type", None)
    if type_name is None:
        raise KeyError(f"config for stage '{stage}' must include a 'type' key")
    if type_name not in _REGISTRY[stage]:
        raise KeyError(
            f"Unknown {stage} type '{type_name}'. "
            f"Registered: {list(_REGISTRY[stage])}"
        )
    cls = _REGISTRY[stage][type_name]
    return cls(**cfg)


def registered(stage: str | None = None) -> dict | list:
    if stage is None:
        return {s: list(v.keys()) for s, v in _REGISTRY.items()}
    return list(_REGISTRY[stage].keys())
