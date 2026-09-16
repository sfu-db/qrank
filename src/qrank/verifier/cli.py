"""Command-line entry point for dataset verification."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

from qrank.config import VerifierConfig
from qrank.llms import LLMBackend, available_backends, load_backend

from .cache import SQLiteCache
from .models import VerificationError, atomic_write_text
from .pipeline import attach_results, collect_questions, dataset_questions, summarize_rows
from .overall import (PROMPT_VERSION as OVERALL_PROMPT_VERSION,
                      PROMPT_VERSION_V3 as OVERALL_PROMPT_VERSION_V3,
                      PROMPT_VERSION_V4 as OVERALL_PROMPT_VERSION_V4,
                      PROMPT_VERSION_V5 as OVERALL_PROMPT_VERSION_V5,
                      PROMPT_VERSION_V6 as OVERALL_PROMPT_VERSION_V6,
                      PROMPT_VERSION_V7 as OVERALL_PROMPT_VERSION_V7,
                      verify_overall_questions)
from .overall_pipeline import attach_overall_results
from .prompts import PROMPT_VERSION, REQUIREMENTS_VERSION, PERSPECTIVES
from .workflow import audit_event

def load_config(args: argparse.Namespace) -> VerifierConfig:
    config = VerifierConfig.from_file(args.config)
    overrides = {field.name: getattr(args, field.name) for field in fields(VerifierConfig)
                 if getattr(args, field.name, None) is not None}
    return replace(config, **overrides)


def make_backend(args: argparse.Namespace, config: VerifierConfig | None = None) -> LLMBackend:
    if config is None:
        config = load_config(args)
    return load_backend(config.backend, **config.backend_options())



def cmd_transform_dataset(args: argparse.Namespace) -> int:
    output = Path(args.output)
    source = Path(args.input).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise VerificationError("Output must be outside the source dataset")
    if (output / "dataset_dict.json").exists():
        raise VerificationError("Output already contains a dataset; choose a new --output directory")
    config = load_config(args)
    batch_size = config.batch_size
    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    dataset, split, questions = dataset_questions(args)
    output.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args, config=config)
    cache_path = args.cache or str(output / "verifier_cache.sqlite")
    cache = SQLiteCache(cache_path)
    audit_path = output / "verifier_trace.jsonl"
    audit_event(audit_path, stage="run", status="started", backend=backend.identity(),
                prompt_version=PROMPT_VERSION,
                source=str(source), split=args.split, batch_size=batch_size)
    total_clusters = sum(len(q.clusters) for q in questions)
    print(f"Prepared {len(questions)} questions, {total_clusters} clusters; "
          f"{sum(not q.schema for q in questions)} questions without context.",
          file=sys.stderr, flush=True)
    rows = []
    done_clusters = 0
    # Group questions so even datasets with one cluster per question fill batches.
    question_batch_size = max(batch_size, args.progress_every, 1)
    started = time.monotonic()
    for start in range(0, len(questions), question_batch_size):
        batch = questions[start:start + question_batch_size]
        rows.extend(collect_questions(batch, backend, cache, batch_size, perspectives=perspectives, audit_path=audit_path))
        done_clusters += sum(len(q.clusters) for q in batch)
        if args.progress_every > 0:
            elapsed = time.monotonic() - started
            print(f"progress questions={start + len(batch)}/{len(questions)} "
                  f"clusters={done_clusters}/{total_clusters} elapsed={elapsed:.1f}s",
                  file=sys.stderr, flush=True)
    report = summarize_rows(rows, perspectives)
    attach_results(dataset, split, questions, rows, args)
    atomic_write_text(output / "semantic_verifier.manifest.json", json.dumps({
        "source": str(source), "split": args.split, "limit": args.limit,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "requirements_version": REQUIREMENTS_VERSION,
        "perspectives": list(perspectives),
        "workflow": "SQL: shared grounded requirements, local block and requirement coverage signals; PrepBench: code perspectives",
        "overall_verdict": False,
        "verification_unit": "distinct stripped candidate text within each agreement cluster",
        "error_policy": "raise immediately; no retries or fallback labels",
        "trace": str(audit_path),
        "backend": backend.identity(),
        "cache": str(cache_path),
        "batch_size": batch_size,
        "annotation_column": "semantic_verifier",
        "task_types": sorted({q.task_type for q in questions}),
        "alignment": "same order and length as candidates; invalid candidates have status=invalid",
        **report,
    }, indent=2))
    print(f"Saved transformed DatasetDict to {output}", flush=True)
    return 0


def cmd_transform_overall_dataset(args: argparse.Namespace) -> int:
    """Save one question response containing overall S/U/C/NA per candidate."""
    output = Path(args.output)
    source = Path(args.input).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise VerificationError("Output must be outside the source dataset")
    if (output / "dataset_dict.json").exists():
        raise VerificationError("Output already contains a dataset; choose a new --output directory")
    config = load_config(args)
    dataset, split, questions = dataset_questions(args)
    output.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args, config=config)
    cache_path = args.cache or str(output / "verifier_cache.sqlite")
    cache = SQLiteCache(cache_path)
    audit_path = output / "verifier_trace.jsonl"
    prompt_version = {"v2": OVERALL_PROMPT_VERSION,
                      "v3": OVERALL_PROMPT_VERSION_V3,
                      "v4": OVERALL_PROMPT_VERSION_V4,
                      "v5": OVERALL_PROMPT_VERSION_V5,
                      "v6": OVERALL_PROMPT_VERSION_V6,
                      "v7": OVERALL_PROMPT_VERSION_V7,
                      }[args.overall_prompt]
    audit_event(audit_path, stage="run", status="started", backend=backend.identity(),
                prompt_version=prompt_version, source=str(source), split=args.split,
                batch_size=config.batch_size, input_models=args.models)
    results = []
    question_batch_size = max(config.batch_size, args.progress_every, 1)
    total = sum(len(q.clusters) for q in questions)
    done = 0
    for start in range(0, len(questions), question_batch_size):
        batch = questions[start:start + question_batch_size]
        results.extend(result for question_results in verify_overall_questions(
            batch, backend, cache, config.batch_size, audit_path=audit_path,
            prompt_variant=args.overall_prompt,
        ) for result in question_results)
        done += sum(len(q.clusters) for q in batch)
        if args.progress_every > 0:
            print(f"progress questions={start + len(batch)}/{len(questions)} "
                  f"candidates={done}/{total}", file=sys.stderr, flush=True)
    counts = attach_overall_results(dataset, split, questions, results, str(output), args.split)
    n_retained = sum(len(row["candidates"]) for row in split)
    atomic_write_text(output / "semantic_verifier_overall.manifest.json", json.dumps({
        "source": str(source), "split": args.split, "limit": args.limit,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "prompt_variant": args.overall_prompt,
        "workflow": "one source-blind overall S/U/C/NA response per nonempty question",
        "verification_unit": "distinct text within each agreement cluster; identical texts across groups share one verdict",
        "input_models": args.models,
        "agreement_rebuild": "valid selected-model submatrix of the input agreement",
        "error_policy": "raise immediately; no retries or fallback labels",
        "n_questions": len(questions), "n_candidates_checked": len(results),
        "n_displayed_candidates": sum(len({c.representative_sql.strip() for c in q.clusters})
                                      for q in questions),
        "n_question_requests": sum(bool(q.clusters) for q in questions),
        "n_model_generations": len({result.cache_key for result in results if result.status == "ok"}),
        "n_retained_candidates": n_retained,
        "n_filtered_invalid_candidates": len(questions) * len(args.models) - n_retained,
        "n_all_invalid_or_no_valid_cluster": sum(not q.clusters for q in questions),
        "label_counts": counts, "trace": str(audit_path),
        "backend": backend.identity(), "cache": str(cache_path),
        "batch_size": config.batch_size,
        "annotation_column": "semantic_verifier_overall",
        "task_types": sorted({q.task_type for q in questions}),
        "alignment": "same order and length as candidates; invalid candidates have status=invalid",
    }, indent=2))
    print(f"Saved overall-verifier DatasetDict to {output}", flush=True)
    return 0



def add_transformers_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0 (overrides config)")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--revision", help="Hugging Face revision or commit (overrides config)")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect QRank semantic-verifier judgments")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("transform-dataset", help="annotate SQL or PrepBench data with a local LLM")
    overall = sub.add_parser("transform-overall-dataset", help="one question prompt with overall S/U/C/NA per candidate")
    for parser in (a, overall):
        parser.add_argument("--config", required=True, help="Path to verifier JSON config; explicit flags override its values")
        parser.add_argument("--input", default=("dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7"
                                                if parser is overall else "dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5"))
        parser.add_argument("--output", default="dataset/processed")
        parser.add_argument("--split", default="test")
        parser.add_argument("--task-type", choices=("auto", "text-to-sql", "prepbench"), default="auto",
                            help="infer from task_type metadata or PrepBench solve(tables) signature")
        parser.add_argument("--prepbench-root", help="input CSV root: <root>/<question_id>/inputs/*.csv")
        parser.add_argument("--schemas", help="schema.json or BIRD tables JSON; otherwise discover under dataset/base")
        parser.add_argument("--db-root", help="directory containing BIRD .sqlite databases (searched recursively)")
        parser.add_argument("--descriptions-root", help="root containing <db_id>/database_description/*.csv; defaults to --db-root")
        parser.add_argument("--sql-dialect", help="SQL dialect; SQLite for automatic BIRD or --db-root discovery")
        parser.add_argument("--limit", type=int, help="process only the first N rows for a pilot")
        parser.add_argument("--backend", choices=available_backends(), help="registered LLM backend name")
        parser.add_argument("--model")
        parser.add_argument("--cache")
        parser.add_argument("--batch-size", type=int, help="generation requests per batch (overrides config; default: 8)")
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--max-tokens", type=int)
        parser.add_argument("--progress-every", type=int, default=10)
        add_transformers_options(parser)
    a.add_argument("--perspective", choices=PERSPECTIVES, help="judge only this perspective; default: all four independently")
    a.set_defaults(func=cmd_transform_dataset)
    overall.add_argument("--models", "--candidate-models", nargs="+", required=True,
                         help="candidate source/model names to retain, in the requested order")
    overall.add_argument("--overall-prompt", choices=("v2", "v3", "v4", "v5", "v6", "v7"), default="v2",
                         help="overall verdict prompt variant (default: v2)")
    overall.set_defaults(func=cmd_transform_overall_dataset)

    return p



def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
