"""CLI for the naive LLM selector baseline."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from qrank.llms import BackendError, load_backend

from .config import load_selector_config
from .pipeline import run_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run naive LLM candidate selector over processed datasets")
    parser.add_argument(
        "--config",
        default="training_configs/selector_naive_qwen3_1p7b.json",
        help="Path to JSON config file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(f"Loading selector config: {args.config}", file=sys.stderr, flush=True)
        cfg = load_selector_config(args.config)
        print(
            f"Loading backend={cfg.backend} model={cfg.model} revision={cfg.revision} "
            f"device={cfg.device} dtype={cfg.dtype}",
            file=sys.stderr,
            flush=True,
        )
        backend = load_backend(
            cfg.backend,
            model=cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            revision=cfg.revision,
        )
        print(
            f"Running dataset={cfg.dataset.name} split={cfg.dataset.split} path={cfg.dataset.path}",
            file=sys.stderr,
            flush=True,
        )
        manifest = run_target(cfg, cfg.dataset, backend)
        print("Selection run completed.", file=sys.stderr, flush=True)
        print(json.dumps({"output_dir": cfg.output_dir, "run": manifest}, indent=2))
        return 0
    except (FileNotFoundError, ValueError, BackendError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
