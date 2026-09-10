"""Run the naive selector baseline over dataset rows."""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import load_from_disk

from qrank.llms import BackendError, LLMBackend

from .config import SelectorConfig, SelectorDatasetTarget
from .prompts import SYSTEM_PROMPT, selector_user_prompt


@dataclass(frozen=True)
class SelectionResult:
    question_id: str
    status: str
    selected_index: int | None
    confidence: float
    reason: str
    valid_indices: tuple[int, ...]
    raw_response: str
    requested_oracle: bool
    used_oracle: bool
    final_source: str
    is_correct: bool | None


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(obj, dict):
        raise ValueError("response JSON must be an object")
    return obj


def _valid_candidate_indices(candidates: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    indices = []
    for i, candidate in enumerate(candidates):
        if bool(candidate.get("invalid_mask", False)):
            continue
        if not str(candidate.get("text", "")).strip():
            continue
        indices.append(i)
    return tuple(indices)


def _parse_selection(raw: str, total_candidates: int, valid_indices: tuple[int, ...]) -> tuple[int | None, float, str]:
    obj = _extract_json_object(raw)
    if "selected_index" not in obj:
        raise ValueError("missing selected_index")
    selected_raw = obj["selected_index"]
    if selected_raw is None or (isinstance(selected_raw, str) and selected_raw.strip().lower() == "none"):
        selected_index = None
    else:
        selected_index = int(selected_raw)
        if selected_index < 0 or selected_index >= total_candidates:
            raise ValueError("selected_index is out of range")
        if selected_index not in valid_indices:
            raise ValueError("selected_index points to invalid/empty candidate")

    confidence_raw = obj.get("confidence", 0.0)
    confidence = float(confidence_raw)
    if confidence < 0 or confidence > 1:
        confidence = min(max(confidence, 0.0), 1.0)
    reason = str(obj.get("reason", "")).strip()
    return selected_index, confidence, reason


def _fallback(question_id: str, reason: str, valid_indices: tuple[int, ...], raw_response: str) -> SelectionResult:
    if not valid_indices:
        return SelectionResult(question_id, "no_valid_candidates", None, 0.0, reason,
                               valid_indices, raw_response, False, False, "none", None)
    return SelectionResult(question_id, "fallback", valid_indices[0], 0.0, reason,
                           valid_indices, raw_response, False, False, "fallback", None)


def _oracle_budget(n_rows: int, ratio: float, count: int | None) -> int:
    ratio_budget = int(n_rows * ratio)
    if count is None:
        return ratio_budget
    return min(ratio_budget, count)


def _print_progress(done: int, total: int, started: float, final: bool = False) -> None:
    if total <= 0:
        return
    width = 28
    ratio = done / total
    filled = min(width, int(width * ratio))
    bar = ("#" * filled) + ("-" * (width - filled))
    elapsed = time.monotonic() - started
    line = f"selector progress [{bar}] {done}/{total} ({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s"
    if final or not sys.stderr.isatty():
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, file=sys.stderr, end="\r", flush=True)


def _label_correctness(row: Mapping[str, Any], selected_index: int | None, source: str) -> bool | None:
    if source == "oracle":
        return True
    labels = row.get("labels")
    if not isinstance(labels, list):
        return None
    if selected_index is None or selected_index < 0 or selected_index >= len(labels):
        return None
    return bool(labels[selected_index])


def _apply_oracle_policy(
    rows: Sequence[Mapping[str, Any]], results: Sequence[SelectionResult],
    oracle_confidence_threshold: float, oracle_budget_ratio: float,
    oracle_budget_count: int | None,
) -> list[SelectionResult]:
    budget = _oracle_budget(len(results), oracle_budget_ratio, oracle_budget_count)
    remaining = budget
    adjusted: list[SelectionResult] = []
    for row, result in zip(rows, results, strict=True):
        requested_oracle = result.selected_index is None or result.confidence < oracle_confidence_threshold
        selected_index = result.selected_index
        status = result.status
        reason = result.reason
        used_oracle = False
        final_source = "llm" if status == "ok" else result.final_source

        if requested_oracle and remaining > 0:
            remaining -= 1
            used_oracle = True
            selected_index = None
            final_source = "oracle"
            if status == "ok":
                status = "oracle"
        elif requested_oracle and selected_index is None:
            if result.valid_indices:
                selected_index = result.valid_indices[0]
                final_source = "fallback"
                status = "fallback"
                reason = f"oracle requested but budget exhausted; {reason}"
            else:
                final_source = "none"

        is_correct = _label_correctness(row, selected_index, final_source)
        adjusted.append(SelectionResult(
            question_id=result.question_id,
            status=status,
            selected_index=selected_index,
            confidence=result.confidence,
            reason=reason,
            valid_indices=result.valid_indices,
            raw_response=result.raw_response,
            requested_oracle=requested_oracle,
            used_oracle=used_oracle,
            final_source=final_source,
            is_correct=is_correct,
        ))
    return adjusted


def run_selection(
    rows: Sequence[Mapping[str, Any]], backend: LLMBackend, batch_size: int,
    oracle_confidence_threshold: float = 0.6, oracle_budget_ratio: float = 0.2,
    oracle_budget_count: int | None = None,
    show_progress: bool = True, progress_every: int = 1,
) -> list[SelectionResult]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    requests = []
    valid_map = {}
    question_ids = []
    for i, row in enumerate(rows):
        qid = str(row.get("question_id", i))
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError(f"question_id={qid}: candidates must be a list")
        valid = _valid_candidate_indices(candidates)
        valid_map[i] = valid
        question_ids.append(qid)
        requests.append(
            {
                "question_id": qid,
                "cluster_id": "selector",
                "system": SYSTEM_PROMPT,
                "user": selector_user_prompt(row),
            }
        )

    responses: list[str | Exception] = [""] * len(rows)
    started = time.monotonic()
    total_requests = len(requests)
    total_batches = ((total_requests + batch_size - 1) // batch_size) if total_requests else 0
    for batch_idx, start in enumerate(range(0, len(requests), batch_size), start=1):
        stop = start + batch_size
        batch = requests[start:stop]
        try:
            batch_responses = backend.generate_batch(batch)
            for i, raw in enumerate(batch_responses, start=start):
                responses[i] = raw
        except (BackendError, OSError, ValueError) as exc:
            for i in range(start, min(stop, len(requests))):
                responses[i] = exc
        if show_progress and total_requests > 0:
            is_tick = (batch_idx % progress_every) == 0 or batch_idx == total_batches
            if is_tick:
                _print_progress(min(stop, total_requests), total_requests, started,
                                final=(batch_idx == total_batches))

    results: list[SelectionResult] = []
    for i, row in enumerate(rows):
        qid = question_ids[i]
        raw = responses[i]
        total = len(row.get("candidates", []))
        valid = valid_map[i]
        if isinstance(raw, Exception):
            results.append(_fallback(qid, f"backend error: {type(raw).__name__}: {raw}", valid, ""))
            continue
        try:
            selected, confidence, reason = _parse_selection(raw, total, valid)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            results.append(_fallback(qid, f"parse failure: {type(exc).__name__}: {exc}", valid, raw))
            continue
        results.append(SelectionResult(
            qid, "ok", selected, confidence, reason, valid, raw,
            False, False, "llm", None,
        ))
    return _apply_oracle_policy(
        rows,
        results,
        oracle_confidence_threshold=oracle_confidence_threshold,
        oracle_budget_ratio=oracle_budget_ratio,
        oracle_budget_count=oracle_budget_count,
    )


def _row_output(row: Mapping[str, Any], result: SelectionResult) -> dict[str, Any]:
    selected_text = None
    candidates = row.get("candidates", [])
    if result.selected_index is not None and 0 <= result.selected_index < len(candidates):
        selected_text = str(candidates[result.selected_index].get("text", ""))
    return {
        "question_id": result.question_id,
        "status": result.status,
        "selected_index": result.selected_index,
        "selected_text": selected_text,
        "confidence": result.confidence,
        "reason": result.reason,
        "valid_indices": list(result.valid_indices),
        "requested_oracle": result.requested_oracle,
        "used_oracle": result.used_oracle,
        "final_source": result.final_source,
        "is_correct": result.is_correct,
    }


def _compute_metrics(results: Sequence[SelectionResult]) -> dict[str, Any]:
    labeled = [r for r in results if r.is_correct is not None]
    llm_rows = [r for r in labeled if r.final_source in {"llm", "fallback"}]
    e2e_rows = [r for r in labeled if r.final_source in {"llm", "fallback", "oracle"}]

    llm_acc = (sum(bool(r.is_correct) for r in llm_rows) / len(llm_rows)) if llm_rows else None
    e2e_acc = (sum(bool(r.is_correct) for r in e2e_rows) / len(e2e_rows)) if e2e_rows else None
    oracle_rate = (sum(1 for r in results if r.used_oracle) / len(results)) if results else 0.0

    return {
        "n_rows_with_labels": len(labeled),
        "llm_only_accuracy": llm_acc,
        "llm_only_denominator": len(llm_rows),
        "end_to_end_accuracy": e2e_acc,
        "end_to_end_denominator": len(e2e_rows),
        "oracle_usage_rate": oracle_rate,
        "oracle_calls": sum(1 for r in results if r.used_oracle),
        "oracle_requested": sum(1 for r in results if r.requested_oracle),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_target(config: SelectorConfig, target: SelectorDatasetTarget, backend: LLMBackend) -> dict[str, Any]:
    dataset = load_from_disk(target.path)
    if target.split not in dataset:
        raise ValueError(f"Split {target.split!r} not found in {target.path}")
    split = dataset[target.split]
    if target.limit is not None:
        split = split.select(range(min(target.limit, len(split))))

    rows = [split[i] for i in range(len(split))]
    print(f"Prepared {len(rows)} rows for selection.", file=sys.stderr, flush=True)
    results = run_selection(
        rows,
        backend,
        config.batch_size,
        oracle_confidence_threshold=config.oracle_confidence_threshold,
        oracle_budget_ratio=config.oracle_budget_ratio,
        oracle_budget_count=config.oracle_budget_count,
        show_progress=config.show_progress,
        progress_every=config.progress_every,
    )

    output_dir = Path(config.output_dir) / target.name
    if output_dir.exists() and not config.overwrite and any(output_dir.iterdir()):
        raise ValueError(f"Output directory {output_dir} is not empty; set overwrite=true")
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = [_row_output(row, result) for row, result in zip(rows, results, strict=True)]
    _write_jsonl(output_dir / "selections.jsonl", selection_rows)

    status_counts = {
        "ok": sum(1 for r in results if r.status == "ok"),
        "oracle": sum(1 for r in results if r.status == "oracle"),
        "fallback": sum(1 for r in results if r.status == "fallback"),
        "no_valid_candidates": sum(1 for r in results if r.status == "no_valid_candidates"),
    }
    metrics = _compute_metrics(results)
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"name": target.name, "path": target.path, "split": target.split, "limit": target.limit},
        "backend": backend.identity(),
        "batch_size": config.batch_size,
        "oracle_policy": {
            "confidence_threshold": config.oracle_confidence_threshold,
            "budget_ratio": config.oracle_budget_ratio,
            "budget_count": config.oracle_budget_count,
        },
        "n_rows": len(rows),
        "status_counts": status_counts,
        "metrics": metrics,
        "selection_file": "selections.jsonl",
    }
    (output_dir / "selector.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
