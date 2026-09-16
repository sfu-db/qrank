"""Shared LLM configuration data model; run settings live in JSON files."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """Model and generation settings supplied explicitly by each run."""

    backend: str
    model: str
    device: str
    dtype: str
    temperature: float
    max_tokens: int
    revision: str
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend_kwargs, dict) or any(
            not isinstance(key, str) for key in self.backend_kwargs
        ):
            raise ValueError("backend_kwargs must be an object with string keys")
        reserved = set(self.__dataclass_fields__) & self.backend_kwargs.keys()
        if reserved:
            raise ValueError(f"backend_kwargs cannot override shared settings: {sorted(reserved)}")
        for name in ("backend", "model", "device", "dtype", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be a finite non-negative number")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Validate a complete configuration without implicit model defaults."""
        if not isinstance(data, dict):
            raise ValueError("LLM config must be a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Invalid LLM config: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> LLMConfig:
        """Read a JSON config; relative paths resolve from the working directory."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def backend_options(self) -> dict[str, Any]:
        """Return the keyword arguments used to construct the backend."""
        options = asdict(self)
        options.pop("backend")
        options.update(options.pop("backend_kwargs"))
        return options


@dataclass(frozen=True)
class VerifierConfig(LLMConfig):
    """LLM settings plus verifier runtime batching."""

    batch_size: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def backend_options(self) -> dict[str, Any]:
        options = super().backend_options()
        options.pop("batch_size")
        return options
