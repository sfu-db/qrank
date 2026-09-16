"""Utilities for joining PrepBench evaluation matrices with task/code exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from datasets import Features, List, Value


TRANSFORMED_FEATURES = Features({
    "question": Value("string"),
    "task_type": Value("string"),
    "labels": List(Value("bool")),
    "candidates": List({
        "source": Value("string"),
        "text": Value("string"),
        "invalid_mask": Value("bool"),
        "features": List(Value("float64")),
    }),
})


def load_exports(path: str | Path) -> dict[str, dict[str, Any]]:
    """Index a PrepBench JSONL task/code export by case ID."""
    source = Path(path)
    exports: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["question_id"])
            if case_id in exports:
                raise ValueError(f"Duplicate PrepBench case {case_id!r} in {source}")
            exports[case_id] = row
    return exports


def transform_row(
    row: Mapping[str, Any], exports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Add task text and candidates to one matrix row in matrix-model order."""
    case_id = str(row["case_id"])
    if case_id not in exports:
        raise ValueError(f"PrepBench case {case_id!r} is missing from the JSONL export")
    export = exports[case_id]
    question = export.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"PrepBench case {case_id!r} has no task text")
    export_candidates = export.get("candidates")
    if not isinstance(export_candidates, list):
        raise ValueError(f"PrepBench case {case_id!r} has no candidate list")
    by_model: dict[str, Mapping[str, Any]] = {}
    for candidate in export_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"PrepBench case {case_id!r} has a non-object candidate")
        model = candidate.get("source")
        if not isinstance(model, str) or not model:
            raise ValueError(f"PrepBench case {case_id!r} has a candidate without a source")
        if model in by_model:
            raise ValueError(f"PrepBench case {case_id!r} has duplicate model {model!r}")
        by_model[model] = candidate

    models = row["models"]
    n = len(models)
    if len(set(models)) != n:
        raise ValueError(f"PrepBench case {case_id!r} has duplicate matrix models")
    agreement = row["agreement"]
    if len(agreement) != n or any(len(values) != n for values in agreement):
        raise ValueError(f"PrepBench case {case_id!r}: agreement shape mismatch")
    features = row["answer_features"]
    if len(features) != n or any(len(values) != len(row["feature_names"]) for values in features):
        raise ValueError(f"PrepBench case {case_id!r}: answer feature shape mismatch")
    labels = row["labels"]
    if len(labels) != n or any(label not in (None, 0, 1) for label in labels):
        raise ValueError(f"PrepBench case {case_id!r}: expected one binary label per model")

    invalid = set(row.get("invalid_models") or [])
    candidates = []
    for i, model in enumerate(models):
        candidate = by_model.get(model)
        if candidate is None and model not in invalid:
            raise ValueError(f"PrepBench case {case_id!r} is missing code for model {model!r}")
        text = candidate.get("text") if candidate is not None else ""
        if not isinstance(text, str):
            raise ValueError(f"PrepBench case {case_id!r}, model {model!r}: text must be a string")
        candidates.append({
            "source": model,
            "text": text,
            "invalid_mask": (
                model in invalid or not text.strip() or bool((candidate or {}).get("invalid_mask", False))
            ),
            "features": [float(value) for value in features[i]],
        })

    return {
        "question": question,
        "task_type": "prepbench",
        "candidates": candidates,
        "labels": [None if label is None else bool(label) for label in labels],
    }


__all__ = ["TRANSFORMED_FEATURES", "load_exports", "transform_row"]
