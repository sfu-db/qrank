"""Register backend classes and lazily import built-in implementations."""
from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from .base import LLMBackend

BackendType = TypeVar("BackendType", bound=type[LLMBackend])
_BACKENDS: dict[str, type[LLMBackend]] = {}
_BACKEND_MODULES = {"transformers": "qrank.llms.transformers"}


def register_backend(name: str) -> Callable[[BackendType], BackendType]:
    """Register an implementation; duplicate names are programming errors."""
    def register(cls: BackendType) -> BackendType:
        if name in _BACKENDS:
            raise ValueError(f"Backend {name!r} is already registered")
        cls.backend_name = name
        _BACKENDS[name] = cls
        return cls

    return register


def available_backends() -> tuple[str, ...]:
    """List names without importing implementations."""
    return tuple(sorted(_BACKEND_MODULES.keys() | _BACKENDS.keys()))


def load_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Import a built-in on demand, then instantiate its registered class.

    Custom backend modules must be imported to run their registration decorator
    before calling this function. Unknown names raise KeyError.
    """
    if name not in _BACKENDS:
        importlib.import_module(_BACKEND_MODULES[name])
    return _BACKENDS[name](**kwargs)
