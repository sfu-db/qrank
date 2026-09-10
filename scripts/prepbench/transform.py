"""Annotate the PrepBench export using the shared semantic-verifier pipeline.

Run from the project root with:
    uv run --no-sync python scripts/prepbench/transform.py --limit 5 \
        --output dataset/processed-prepbench-pilot

All transform-dataset options are accepted; --help lists them. Default paths
are relative to the repository, while explicit paths are relative to the caller.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from qrank.verifier.cli import main as verifier_main


ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    return verifier_main([
        "transform-dataset",
        "--input", str(ROOT / "dataset/raw/prepbench.jsonl"),
        "--output", str(ROOT / "dataset/processed-prepbench"),
        "--task-type", "prepbench",
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
