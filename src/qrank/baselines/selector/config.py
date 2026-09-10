"""Configuration model for the naive selector baseline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qrank import config as qrank_config


@dataclass(frozen=True)
class SelectorDatasetTarget:
    """One dataset split to run candidate selection over."""

    name: str
    path: str
    split: str = "test"
    limit: int | None = None


@dataclass(frozen=True)
class SelectorConfig:
    """Configurable runtime options for the selector baseline."""

    output_dir: str
    dataset: SelectorDatasetTarget
    backend: str = qrank_config.LLM_BACKEND
    model: str = qrank_config.LLM_OPTIONS["model"]
    device: str = qrank_config.LLM_OPTIONS["device"]
    dtype: str = qrank_config.LLM_OPTIONS["dtype"]
    temperature: float = 0.0
    max_tokens: int = 128
    revision: str = qrank_config.LLM_OPTIONS["revision"]
    batch_size: int = 4
    overwrite: bool = False
    oracle_confidence_threshold: float = 0.6
    oracle_budget_ratio: float = 0.2
    oracle_budget_count: int | None = None
    show_progress: bool = True
    progress_every: int = 1


def _as_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return parsed


def _load_target(value: Any) -> SelectorDatasetTarget:
    if not isinstance(value, dict):
        raise ValueError("dataset must be an object")
    name = str(value.get("name", "")).strip()
    path = str(value.get("path", "")).strip()
    split = str(value.get("split", "test")).strip() or "test"
    if not name:
        raise ValueError("dataset.name must be non-empty")
    if not path:
        raise ValueError("dataset.path must be non-empty")
    limit = value.get("limit")
    if limit is not None:
        limit = _as_int(limit, "dataset.limit")
        if limit <= 0:
            raise ValueError("dataset.limit must be positive")
    return SelectorDatasetTarget(name=name, path=path, split=split, limit=limit)


def load_selector_config(path: str | Path) -> SelectorConfig:
    """Load and validate a selector config JSON file."""

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Selector config must be a JSON object")

    output_dir = str(data.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("output_dir must be non-empty")

    batch_size = _as_int(data.get("batch_size", 4), "batch_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    max_tokens = _as_int(data.get("max_tokens", 128), "max_tokens")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    temperature = float(data.get("temperature", 0.0))
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    oracle_confidence_threshold = float(data.get("oracle_confidence_threshold", 0.6))
    if oracle_confidence_threshold < 0 or oracle_confidence_threshold > 1:
        raise ValueError("oracle_confidence_threshold must be in [0, 1]")

    oracle_budget_ratio = float(data.get("oracle_budget_ratio", 0.2))
    if oracle_budget_ratio < 0 or oracle_budget_ratio > 1:
        raise ValueError("oracle_budget_ratio must be in [0, 1]")

    oracle_budget_count = data.get("oracle_budget_count")
    if oracle_budget_count is not None:
        oracle_budget_count = _as_int(oracle_budget_count, "oracle_budget_count")
        if oracle_budget_count < 0:
            raise ValueError("oracle_budget_count must be non-negative")

    progress_every = _as_int(data.get("progress_every", 1), "progress_every")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    show_progress = bool(data.get("show_progress", True))

    overwrite = bool(data.get("overwrite", False))

    return SelectorConfig(
        output_dir=output_dir,
        dataset=_load_target(data.get("dataset")),
        backend=str(data.get("backend", qrank_config.LLM_BACKEND)),
        model=str(data.get("model", qrank_config.LLM_OPTIONS["model"])),
        device=str(data.get("device", qrank_config.LLM_OPTIONS["device"])),
        dtype=str(data.get("dtype", qrank_config.LLM_OPTIONS["dtype"])),
        temperature=temperature,
        max_tokens=max_tokens,
        revision=str(data.get("revision", qrank_config.LLM_OPTIONS["revision"])),
        batch_size=batch_size,
        overwrite=overwrite,
        oracle_confidence_threshold=oracle_confidence_threshold,
        oracle_budget_ratio=oracle_budget_ratio,
        oracle_budget_count=oracle_budget_count,
        show_progress=show_progress,
        progress_every=progress_every,
    )
