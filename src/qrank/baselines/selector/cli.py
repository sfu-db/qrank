"""CLI for the naive LLM selector baseline."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from qrank.llms import BackendError, load_backend

from .config import load_selector_config
from .pipeline import prepare_output_dir, run_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run naive LLM candidate selector over processed datasets")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(f"Loading selector config: {args.config}", file=sys.stderr, flush=True)
        cfg = load_selector_config(args.config)
        prepare_output_dir(cfg, cfg.dataset)
        print(
            f"Loading backend={cfg.backend} model={cfg.model} revision={cfg.revision} "
            f"device={cfg.device} dtype={cfg.dtype}",
            file=sys.stderr,
            flush=True,
        )
        backend = load_backend(cfg.backend, **cfg.backend_options())
        print(
            f"Running dataset={cfg.dataset.name} split={cfg.dataset.split} path={cfg.dataset.path}",
            file=sys.stderr,
            flush=True,
        )
        manifest = run_target(cfg, cfg.dataset, backend)
        print("Selection run completed.", file=sys.stderr, flush=True)
        metrics = manifest["metrics"]
        accuracy = metrics["end_to_end_accuracy"]
        if accuracy is None:
            summary = (f"End-to-end accuracy unavailable: correctness known for "
                       f"{metrics['n_rows_evaluated']}/{manifest['n_rows']} rows. "
                       "Supply ground-truth labels aligned with candidates for unscored rows.")
        else:
            summary = (f"End-to-end accuracy: {accuracy:.2%} over "
                       f"{metrics['end_to_end_denominator']} rows "
                       f"(simulated oracle calls: {metrics['oracle_calls']}).")
        print(summary, file=sys.stderr, flush=True)
        if manifest["budget_sweep"]:
            print("Oracle budget sweep (same model responses; simulated perfect oracle):",
                  file=sys.stderr, flush=True)
            print("Budget ratio | Allowed calls | Used calls | Usage | End-to-end accuracy",
                  file=sys.stderr, flush=True)
            for entry in manifest["budget_sweep"]:
                sweep_metrics = entry["metrics"]
                score = sweep_metrics["end_to_end_accuracy"]
                score_text = "unavailable" if score is None else f"{score:.2%}"
                print(f"{entry['budget_ratio']:.1%} | {entry['effective_budget']} | "
                      f"{sweep_metrics['oracle_calls']} | {sweep_metrics['oracle_usage_rate']:.1%} | "
                      f"{score_text}", file=sys.stderr, flush=True)
        print(f"Results: {cfg.output_dir}/{cfg.dataset.name}/selector.manifest.json",
              file=sys.stderr, flush=True)
        print(json.dumps({"output_dir": cfg.output_dir, "run": manifest}, indent=2))
        return 0
    except (FileNotFoundError, ValueError, BackendError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
