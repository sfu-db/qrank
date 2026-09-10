"""Command-line entry point for dataset verification."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from qrank import config
from qrank.llms import BackendError, LLMBackend, available_backends, load_backend

from .cache import SQLiteCache
from .models import VerificationError, atomic_write_text
from .pipeline import attach_results, collect_questions, dataset_questions, validate_rows
from .prompts import PROMPT_VERSION

def make_backend(args: argparse.Namespace) -> LLMBackend:
    return load_backend(args.backend, model=args.model, device=args.device, dtype=args.dtype,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        revision=args.revision)



def cmd_transform_dataset(args: argparse.Namespace) -> int:
    output = Path(args.output)
    source = Path(args.input).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise VerificationError("Output must be outside the source dataset")
    if (output / "dataset_dict.json").exists():
        raise VerificationError("Output already contains a dataset; choose a new --output directory")
    if args.batch_size < 1:
        raise VerificationError("--batch-size must be positive")
    dataset, split, questions = dataset_questions(args)
    output.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args)
    cache_path = args.cache or str(output / "verifier_cache.sqlite")
    cache = SQLiteCache(cache_path)
    total_clusters = sum(len(q.clusters) for q in questions)
    print(f"Prepared {len(questions)} questions, {total_clusters} clusters; "
          f"{sum(not q.schema for q in questions)} questions without context.",
          file=sys.stderr, flush=True)
    rows = []
    done_clusters = 0
    # Group questions so even datasets with one cluster per question fill batches.
    question_batch_size = max(args.batch_size, args.progress_every, 1)
    started = time.monotonic()
    for start in range(0, len(questions), question_batch_size):
        batch = questions[start:start + question_batch_size]
        rows.extend(collect_questions(batch, backend, cache, args.retries,
                                      args.retry_delay, args.batch_size))
        done_clusters += sum(len(q.clusters) for q in batch)
        if args.progress_every > 0:
            elapsed = time.monotonic() - started
            print(f"progress questions={start + len(batch)}/{len(questions)} "
                  f"clusters={done_clusters}/{total_clusters} elapsed={elapsed:.1f}s",
                  file=sys.stderr, flush=True)
    attach_results(dataset, split, questions, rows, args)
    report = validate_rows(rows)
    atomic_write_text(output / "semantic_verifier.manifest.json", json.dumps({
        "source": str(source), "split": args.split, "limit": args.limit,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "backend": backend.identity(),
        "cache": str(cache_path),
        "batch_size": args.batch_size,
        "annotation_column": "semantic_verifier",
        "task_types": sorted({q.task_type for q in questions}),
        "alignment": "same order and length as candidates; invalid candidates have status=invalid",
        **report,
    }, indent=2))
    print(f"Saved transformed DatasetDict to {output}", flush=True)
    return 0



def add_transformers_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=config.LLM_OPTIONS["device"], help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default=config.LLM_OPTIONS["dtype"])
    parser.add_argument("--revision", default=config.LLM_OPTIONS["revision"], help="Hugging Face revision or commit")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect frozen QRank semantic-verifier S/U/C measurements")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("transform-dataset", help="annotate SQL or PrepBench data with local Qwen3 via Transformers")
    a.add_argument("--input", default="dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5")
    a.add_argument("--output", default="dataset/processed")
    a.add_argument("--split", default="test")
    a.add_argument("--task-type", choices=("auto", "text-to-sql", "prepbench"), default="auto",
                   help="infer from task_type metadata or PrepBench solve(tables) signature")
    a.add_argument("--prepbench-root", help="input CSV root: <root>/<question_id>/inputs/*.csv")
    a.add_argument("--schemas", help="schema.json or BIRD tables JSON; otherwise discover under dataset/base")
    a.add_argument("--db-root", help="directory containing BIRD .sqlite databases (searched recursively)")
    a.add_argument("--limit", type=int, help="process only the first N rows for a pilot")
    a.add_argument("--backend", choices=available_backends(), default=config.LLM_BACKEND,
                   help="registered LLM backend name")
    a.add_argument("--model", default=config.LLM_OPTIONS["model"])
    a.add_argument("--cache")
    a.add_argument("--batch-size", type=int, default=8, help="candidate prompts per generation batch")
    a.add_argument("--temperature", type=float, default=config.LLM_OPTIONS["temperature"])
    a.add_argument("--max-tokens", type=int, default=config.LLM_OPTIONS["max_tokens"])
    a.add_argument("--retries", type=int, default=2)
    a.add_argument("--retry-delay", type=float, default=1.0)
    a.add_argument("--progress-every", type=int, default=10)
    add_transformers_options(a)
    a.set_defaults(func=cmd_transform_dataset)

    return p



def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (VerificationError, BackendError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
