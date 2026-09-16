"""Command-line runner for cluster-level SQL judging."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, DatasetDict, load_from_disk

from qrank.config import VerifierConfig
from qrank.llms import BackendError, load_backend
from qrank.verifier.context import build_context, load_database_descriptions, load_database_schemas

from .pipeline import judge_rows, select_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a model to choose the best SQL agreement cluster")
    parser.add_argument("--config", required=True, help="JSON model/backend config")
    parser.add_argument(
        "--models", "--model", dest="models", nargs="+", required=True,
        help="candidate source/model names to retain when rebuilding agreement",
    )
    parser.add_argument("--input", default="dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--schemas", help="database-to-DDL schema.json or BIRD tables JSON")
    parser.add_argument(
        "--db-root",
        help="optional directory used only for schema/description context; candidate SQL is never executed",
    )
    parser.add_argument("--descriptions-root", help="root containing database_description CSV files")
    parser.add_argument("--sql-dialect", default="SQLite", help="SQL dialect for prompt context")
    parser.add_argument("--limit", type=int, help="judge only the first N rows")
    parser.add_argument("--column", default="qwen_judge",
                        help="name of the new dataset column (default: qwen_judge)")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    output = Path(args.output)
    source = Path(args.input).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise ValueError("output must be outside the input dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    if not isinstance(args.column, str) or not args.column.strip():
        raise ValueError("--column must be non-empty")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicates")

    config = VerifierConfig.from_file(args.config)
    if config.backend != "vllm":
        raise ValueError("judge config must use backend='vllm'")
    dataset = load_from_disk(args.input)
    if args.split not in dataset:
        raise ValueError(f"split {args.split!r} not found in {args.input}")
    split = dataset[args.split]
    if args.limit is not None:
        split = split.select(range(min(args.limit, len(split))))
    rows = [select_models(split[index], args.models) for index in range(len(split))]
    split = Dataset.from_list(rows, features=split.features)
    n_valid_candidates = sum(len(row["candidates"]) for row in rows)

    schemas = load_database_schemas(args)
    descriptions = load_database_descriptions(args)
    dialect = args.sql_dialect or ("SQLite" if schemas and (args.db_root or not args.schemas) else None)
    contexts = [build_context(
        row, task_type="text-to-sql", schemas=schemas, descriptions=descriptions,
        sql_dialect=dialect,
    ).context for row in rows]

    print(f"Loading {config.model} with backend={config.backend}", file=sys.stderr, flush=True)
    backend = load_backend(config.backend, **config.backend_options())
    print(
        f"Judging {len(rows)} questions using {n_valid_candidates} valid candidates "
        f"from {len(args.models)} requested models",
        file=sys.stderr, flush=True,
    )
    results = judge_rows(rows, contexts, backend, batch_size=config.batch_size)

    annotations = [asdict(result) for result in results]
    annotated_split = split.add_column(args.column, annotations)
    annotated = DatasetDict({name: value for name, value in dataset.items()})
    annotated[args.split] = annotated_split
    annotated.save_to_disk(str(output))
    evaluated = [result for result in results if result.is_correct is not None]
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "split": args.split,
        "limit": args.limit,
        "n_rows": len(results),
        "n_selected": sum(result.selected_candidate_index is not None for result in results),
        "n_none": sum(result.status == "none" for result in results),
        "n_no_valid_candidates": sum(result.status == "no_valid_candidates" for result in results),
        "n_valid_candidates": n_valid_candidates,
        "n_filtered_invalid_candidates": len(rows) * len(args.models) - n_valid_candidates,
        "n_evaluated": len(evaluated),
        "accuracy": (sum(bool(result.is_correct) for result in evaluated) / len(evaluated)
                     if evaluated else None),
        "backend": backend.identity(),
        "input_models": args.models,
        "batch_size": config.batch_size,
        "annotation_column": args.column,
        "selection_unit": "first representative of each selected-model agreement cluster",
        "agreement_rebuild": "valid selected-model submatrix of the input agreement",
    }
    (output / "judge.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (BackendError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
