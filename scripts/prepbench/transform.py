"""Merge a PrepBench matrix dataset and JSONL export into a Hugging Face dataset.

From the project root:
    uv run --no-sync python scripts/prepbench/transform.py

All input splits are converted by default. Use --split test or --limit 5 for
a subset. Default paths are relative to the repository; explicit paths are
relative to the caller. JSONL supplies task text and code; the saved matrix
supplies model order, agreement, labels, and validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from datasets import Dataset, DatasetDict, Features, List, Value, load_from_disk


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "dataset/raw/matrix_kimi-k2_glm-4.7_qwen3-235b-a22b_"
    "deepseek-v3.2_devstral-2512_prepbench_m5"
)
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


def load_exports(path: Path) -> dict[str, dict]:
    """Index JSONL task text and code by case ID."""
    exports = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["question_id"])
            if case_id in exports:
                raise ValueError(f"Duplicate PrepBench case {case_id!r} in {path}")
            exports[case_id] = row
    return exports


def transform_row(row: dict, exports: dict[str, dict]) -> dict:
    """Build candidate fields in matrix model order, preserving label alignment."""
    case_id = str(row["case_id"])
    if case_id not in exports:
        raise ValueError(f"PrepBench case {case_id!r} is missing from the JSONL export")
    export = exports[case_id]
    question = export.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"PrepBench case {case_id!r} has no task text")
    by_model = {}
    for candidate in export["candidates"]:
        model = candidate["source"]
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

    # Dataset.map keeps the matrix fields; only these fields are added/replaced.
    return {
        "question": question,
        "task_type": "prepbench",
        "candidates": candidates,
        "labels": [None if label is None else bool(label) for label in labels],
    }


def transform_dataset(
    dataset: DatasetDict, exports: dict[str, dict], *,
    split: str | None = None, limit: int | None = None,
) -> DatasetDict:
    """Convert selected splits while retaining the original matrix columns."""
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if split is not None and split not in dataset:
        raise ValueError(f"Unknown split {split!r}; available splits: {list(dataset)}")

    result = DatasetDict()
    for name in [split] if split is not None else dataset:
        source = dataset[name]
        if limit is not None:
            source = source.select(range(min(limit, len(source))))
        features = Features({**source.features, **TRANSFORMED_FEATURES})
        # Dataset.map skips empty splits, so construct their output schema explicitly.
        if not len(source):
            result[name] = Dataset.from_dict({key: [] for key in features}, features=features)
            continue
        result[name] = source.map(
            transform_row,
            fn_kwargs={"exports": exports},
            features=features,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"Prepare {name}",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="saved matrix DatasetDict")
    parser.add_argument("--prepbench-jsonl", type=Path, default=ROOT / "dataset/raw/prepbench.jsonl",
                        help="question/code export joined by case ID and model")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/processed-prepbench")
    parser.add_argument("--split", help="convert only this split (default: all splits)")
    parser.add_argument("--limit", type=int, help="convert the first N rows per selected split")
    args = parser.parse_args(argv)
    try:
        source, output = args.input.resolve(), args.output.resolve()
        if output == source or source in output.parents:
            raise ValueError("Output must be outside the source dataset")
        if output.exists():
            raise ValueError(f"Output already exists: {output}; choose a new --output directory")
        dataset = load_from_disk(str(source))
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Input must be a saved Hugging Face DatasetDict")
        exports = load_exports(args.prepbench_jsonl)
        result = transform_dataset(dataset, exports, split=args.split, limit=args.limit)
        # Empty splits need an Arrow shard to remain loadable with datasets 4.8.
        result.save_to_disk(str(output), num_shards={name: 1 for name in result if not len(result[name])})
        print(f"Saved Hugging Face dataset to {output}: " + ", ".join(
            f"{name}={len(split)} rows" for name, split in result.items()))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
