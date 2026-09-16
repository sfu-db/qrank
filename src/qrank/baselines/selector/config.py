"""Configuration model for the naive selector baseline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from qrank.config import LLMConfig


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
    backend: str
    model: str
    device: str
    dtype: str
    temperature: float
    max_tokens: int
    revision: str
    batch_size: int = 4
    overwrite: bool = False
    oracle_budget_ratio: float = 0.2
    oracle_budget_count: int | None = None
    oracle_budget_ratios: tuple[float, ...] = ()
    show_progress: bool = True
    progress_every: int = 20
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def backend_options(self) -> dict[str, Any]:
        return LLMConfig.from_dict({
            item.name: getattr(self, item.name) for item in fields(LLMConfig)
        }).backend_options()


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

    llm = LLMConfig.from_dict({field.name: data[field.name] for field in fields(LLMConfig)
                               if field.name in data})

    output_dir = str(data.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("output_dir must be non-empty")

    batch_size = _as_int(data.get("batch_size", 4), "batch_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if "oracle_confidence_threshold" in data:
        raise ValueError("oracle_confidence_threshold has been removed; remove it from the config. "
                         "Oracle budgets now prioritize deferrals, then lowest-confidence selections.")

    oracle_budget_ratio = float(data.get("oracle_budget_ratio", 0.2))
    if not 0 <= oracle_budget_ratio <= 1:
        raise ValueError("oracle_budget_ratio must be in [0, 1]")

    oracle_budget_count = data.get("oracle_budget_count")
    if oracle_budget_count is not None:
        oracle_budget_count = _as_int(oracle_budget_count, "oracle_budget_count")
        if oracle_budget_count < 0:
            raise ValueError("oracle_budget_count must be non-negative")

    sweep = data.get("oracle_budget_ratios", [])
    if not isinstance(sweep, list) or any(isinstance(value, bool) for value in sweep):
        raise ValueError("oracle_budget_ratios must be a list of numbers in [0, 1]")
    try:
        oracle_budget_ratios = tuple(dict.fromkeys(float(value) for value in sweep))
    except (TypeError, ValueError) as exc:
        raise ValueError("oracle_budget_ratios must be a list of numbers in [0, 1]") from exc
    if any(not 0 <= ratio <= 1 for ratio in oracle_budget_ratios):
        raise ValueError("oracle_budget_ratios must be in [0, 1]")

    progress_every = _as_int(data.get("progress_every", 20), "progress_every")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    show_progress = bool(data.get("show_progress", True))

    overwrite = bool(data.get("overwrite", False))

    return SelectorConfig(
        output_dir=output_dir,
        dataset=_load_target(data.get("dataset")),
        **asdict(llm),
        batch_size=batch_size,
        overwrite=overwrite,
        oracle_budget_ratio=oracle_budget_ratio,
        oracle_budget_count=oracle_budget_count,
        oracle_budget_ratios=oracle_budget_ratios,
        show_progress=show_progress,
        progress_every=progress_every,
    )
