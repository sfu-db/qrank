"""Naive LLM-based candidate selector baseline."""

from .cli import main
from .config import SelectorConfig, SelectorDatasetTarget, load_selector_config
from .pipeline import SelectionResult, run_target

__all__ = [
    "SelectorConfig",
    "SelectorDatasetTarget",
    "SelectionResult",
    "load_selector_config",
    "run_target",
    "main",
]
