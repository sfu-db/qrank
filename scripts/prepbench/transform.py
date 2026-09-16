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
from pathlib import Path
import sys
from typing import Sequence

from datasets import Dataset, DatasetDict, Features, load_from_disk

from qrank.dataset.prepbench import TRANSFORMED_FEATURES, load_exports, transform_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / (
    "dataset/raw/matrix_kimi-k2_glm-4.7_qwen3-235b-a22b_"
    "deepseek-v3.2_devstral-2512_prepbench_m5"
)
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
