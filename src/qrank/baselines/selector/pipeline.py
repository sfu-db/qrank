"""Run the naive selector baseline over dataset rows."""
from __future__ import annotations

import datetime as dt
import json
import math
import random
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
    has_labels: bool = False


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    start = text.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def _valid_candidate_indices(candidates: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    indices = []
    for i, candidate in enumerate(candidates):
        if bool(candidate.get("invalid_mask", False)):
            continue
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            continue
        indices.append(i)
    return tuple(indices)


def _parse_selection(raw: str, valid_indices: tuple[int, ...]) -> tuple[int | None, float, str]:
    obj = _extract_json_object(raw)
    if "selected_index" not in obj:
        raise ValueError("missing selected_index")
    selected_raw = obj["selected_index"]
    if selected_raw is None or (isinstance(selected_raw, str) and selected_raw.strip().lower() == "none"):
        selected_index = None
    else:
        if isinstance(selected_raw, bool) or not isinstance(selected_raw, (int, str)):
            raise ValueError("selected_index must be an integer or null")
        selected_index = int(selected_raw)
        if selected_index not in valid_indices:
            raise ValueError("selected_index must identify a valid candidate")

    confidence_raw = obj.get("confidence", 0.0)
    confidence = float(confidence_raw)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    confidence = min(max(confidence, 0.0), 1.0)
    reason = str(obj.get("reason", "")).strip()
    return selected_index, confidence, reason


def _fallback(
    question_id: str, reason: str, valid_indices: tuple[int, ...], raw_response: str,
    has_labels: bool,
) -> SelectionResult:
    if not valid_indices:
        return SelectionResult(question_id, "no_valid_candidates", None, 0.0, reason,
                               valid_indices, raw_response, False, False, "none", None, has_labels)
    return SelectionResult(question_id, "fallback", valid_indices[0], 0.0, reason,
                           valid_indices, raw_response, False, False, "fallback", None, has_labels)


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
    if not sys.stderr.isatty():
        print(line, file=sys.stderr, flush=True)
        return
    if final:
        sys.stderr.write("\r" + line + "\n")
    else:
        sys.stderr.write("\r" + line)
    sys.stderr.flush()


def _validate_labels(row: Mapping[str, Any]) -> bool:
    labels = row.get("labels")
    if labels is None:
        return False
    if (not isinstance(labels, list)
            or len(labels) != len(row.get("candidates", []))
            or any(label is not None and not isinstance(label, bool) for label in labels)):
        raise ValueError(f"question_id={row.get('question_id')}: labels must be a list of "
                         "booleans or nulls aligned with candidates")
    return any(label is not None for label in labels)


def _label_correctness(row: Mapping[str, Any], selected_index: int | None, source: str) -> bool | None:
    if source == "oracle":
        return True
    if source == "none":
        # Returning no answer is an end-to-end failure, even without labels.
        return False
    labels = row.get("labels")
    if labels is None:
        return None
    return labels[selected_index]


def _apply_oracle_policy(
    rows: Sequence[Mapping[str, Any]], results: Sequence[SelectionResult],
    oracle_budget_ratio: float,
    oracle_budget_count: int | None,
) -> list[SelectionResult]:
    budget = _oracle_budget(len(results), oracle_budget_ratio, oracle_budget_count)
    # Deferrals have no selected answer whose confidence can be compared.
    # Preserve dataset order among deferrals and among equal-confidence answers.
    ranked_indices = sorted(range(len(results)), key=lambda i: (
        results[i].selected_index is not None,
        results[i].confidence if results[i].selected_index is not None else 0.0,
        i,
    ))
    oracle_indices = set(ranked_indices[:budget])
    adjusted: list[SelectionResult] = []
    for i, (row, result) in enumerate(zip(rows, results, strict=True)):
        requested_oracle = result.selected_index is None
        selected_index = result.selected_index
        status = result.status
        reason = result.reason
        used_oracle = False
        final_source = "llm" if status == "ok" else result.final_source

        if i in oracle_indices:
            used_oracle = True
            selected_index = None
            final_source = "oracle"
            status = "oracle"
        elif requested_oracle:
            if result.valid_indices:
                selected_index = result.valid_indices[0]
                final_source = "fallback"
                status = "fallback"
                reason = f"oracle requested but budget exhausted; {reason}"
            elif row.get("candidates"):
                # Seed independently per row so policy replay and budget sweeps
                # retain the same random fallback without touching global RNG state.
                rng = random.Random(f"selector-fallback-v1:{i}:{result.question_id}")
                selected_index = rng.randrange(len(row["candidates"]))
                final_source = "fallback"
                status = "fallback"
                reason = f"oracle requested but budget exhausted; randomly selected from all candidates; {reason}"
            else:
                final_source = "none"
                status = "no_valid_candidates"

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
            has_labels=result.has_labels,
        ))
    return adjusted


def run_selection(
    rows: Sequence[Mapping[str, Any]], backend: LLMBackend, batch_size: int,
    oracle_budget_ratio: float = 0.2,
    oracle_budget_count: int | None = None,
    show_progress: bool = True, progress_every: int = 1,
) -> list[SelectionResult]:
    if not 0 <= oracle_budget_ratio <= 1:
        raise ValueError("oracle_budget_ratio must be in [0, 1]")
    if oracle_budget_count is not None and (type(oracle_budget_count) is not int or oracle_budget_count < 0):
        raise ValueError("oracle_budget_count must be a non-negative integer")
    results = _generate_selections(rows, backend, batch_size, show_progress, progress_every)
    return _apply_oracle_policy(
        rows, results, oracle_budget_ratio, oracle_budget_count,
    )


def _generate_selections(
    rows: Sequence[Mapping[str, Any]], backend: LLMBackend, batch_size: int,
    show_progress: bool, progress_every: int,
) -> list[SelectionResult]:
    """Generate once; retain null selections so policies can be replayed independently."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    requests = []
    request_row_indices = []
    valid_map = {}
    question_ids = []
    has_labels = []
    for i, row in enumerate(rows):
        qid = str(row.get("question_id", i))
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError(f"question_id={qid}: candidates must be a list")
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise ValueError(f"question_id={qid}: each candidate must be an object")
        has_labels.append(_validate_labels(row))
        valid = _valid_candidate_indices(candidates)
        valid_map[i] = valid
        question_ids.append(qid)
        if not valid:
            continue
        request_row_indices.append(i)
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
    total_batches = (total_requests + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(requests), batch_size), start=1):
        stop = start + batch_size
        batch = requests[start:stop]
        try:
            batch_responses = backend.generate_batch(batch)
            if len(batch_responses) != len(batch):
                raise BackendError(f"Expected {len(batch)} responses, got {len(batch_responses)}")
            for i, raw in zip(request_row_indices[start:stop], batch_responses, strict=True):
                responses[i] = raw
        except (BackendError, OSError, ValueError) as exc:
            for i in request_row_indices[start:stop]:
                responses[i] = exc
        if show_progress:
            is_tick = (batch_idx % progress_every) == 0 or batch_idx == total_batches
            if is_tick:
                _print_progress(min(stop, total_requests), total_requests, started,
                                final=(batch_idx == total_batches))

    results: list[SelectionResult] = []
    for i, qid in enumerate(question_ids):
        raw = responses[i]
        valid = valid_map[i]
        if not valid:
            results.append(_fallback(qid, "no valid candidates; model inference skipped", valid, "",
                                     has_labels[i]))
            continue
        if isinstance(raw, Exception):
            results.append(_fallback(qid, f"backend error: {type(raw).__name__}: {raw}", valid, "",
                                     has_labels[i]))
            continue
        try:
            selected, confidence, reason = _parse_selection(raw, valid)
        except (TypeError, ValueError) as exc:
            results.append(_fallback(qid, f"parse failure: {type(exc).__name__}: {exc}", valid,
                                     raw if isinstance(raw, str) else repr(raw), has_labels[i]))
            continue
        results.append(SelectionResult(
            qid, "ok", selected, confidence, reason, valid, raw,
            False, False, "llm", None, has_labels[i],
        ))
    return results


def _row_output(row: Mapping[str, Any], result: SelectionResult) -> dict[str, Any]:
    selected_text = None
    candidates = row.get("candidates", [])
    if result.selected_index is not None:
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
        "has_labels": result.has_labels,
        "raw_response": result.raw_response,
    }


def _compute_metrics(results: Sequence[SelectionResult]) -> dict[str, Any]:
    evaluated = [r for r in results if r.is_correct is not None]
    llm_rows = [r for r in evaluated if r.final_source in {"llm", "fallback"}]

    llm_acc = (sum(bool(r.is_correct) for r in llm_rows) / len(llm_rows)) if llm_rows else None
    n_correct = sum(r.is_correct is True for r in evaluated)
    complete = bool(results) and len(evaluated) == len(results)
    e2e_acc = n_correct / len(results) if complete else None
    oracle_rate = (sum(1 for r in results if r.used_oracle) / len(results)) if results else 0.0

    return {
        "n_rows_with_labels": sum(r.has_labels for r in results),
        "n_rows_evaluated": len(evaluated),
        "n_rows_missing_correctness": len(results) - len(evaluated),
        "evaluation_status": ("complete" if complete else "partial" if evaluated
                              else "unavailable" if results else "empty"),
        "llm_only_accuracy": llm_acc,
        "llm_only_denominator": len(llm_rows),
        "end_to_end_accuracy": e2e_acc,
        "end_to_end_denominator": len(results),
        "evaluated_accuracy": n_correct / len(evaluated) if evaluated else None,
        "evaluation_coverage": len(evaluated) / len(results) if results else 0.0,
        "oracle_usage_rate": oracle_rate,
        "oracle_calls": sum(1 for r in results if r.used_oracle),
        "oracle_requested": sum(1 for r in results if r.requested_oracle),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_output_dir(config: SelectorConfig, target: SelectorDatasetTarget) -> Path:
    """Check and create the destination before paying for model inference."""
    output_dir = Path(config.output_dir) / target.name
    if output_dir.exists() and not config.overwrite and any(output_dir.iterdir()):
        raise ValueError(f"Output directory {output_dir} is not empty; set overwrite=true")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_target(config: SelectorConfig, target: SelectorDatasetTarget, backend: LLMBackend) -> dict[str, Any]:
    output_dir = prepare_output_dir(config, target)
    dataset = load_from_disk(target.path)
    if target.split not in dataset:
        raise ValueError(f"Split {target.split!r} not found in {target.path}")
    split = dataset[target.split]
    if target.limit is not None:
        split = split.select(range(min(target.limit, len(split))))

    rows = [split[i] for i in range(len(split))]
    print(f"Prepared {len(rows)} rows for selection.", file=sys.stderr, flush=True)
    raw_results = _generate_selections(
        rows,
        backend,
        config.batch_size,
        show_progress=config.show_progress,
        progress_every=config.progress_every,
    )
    results = _apply_oracle_policy(
        rows, raw_results, config.oracle_budget_ratio, config.oracle_budget_count,
    )

    selection_rows = [_row_output(row, result) for row, result in zip(rows, results, strict=True)]
    _write_jsonl(output_dir / "selections.jsonl", selection_rows)

    status_counts = {
        "ok": sum(1 for r in results if r.status == "ok"),
        "oracle": sum(1 for r in results if r.status == "oracle"),
        "fallback": sum(1 for r in results if r.status == "fallback"),
        "no_valid_candidates": sum(1 for r in results if r.status == "no_valid_candidates"),
    }
    metrics = _compute_metrics(results)
    budget_sweep = []
    for index, ratio in enumerate(config.oracle_budget_ratios):
        budget_results = _apply_oracle_policy(
            rows, raw_results, ratio, config.oracle_budget_count,
        )
        selection_file = f"selections.budget_{index}.jsonl"
        _write_jsonl(output_dir / selection_file,
                     (_row_output(row, result) for row, result in zip(rows, budget_results, strict=True)))
        budget_sweep.append({
            "budget_ratio": ratio,
            "effective_budget": _oracle_budget(len(rows), ratio, config.oracle_budget_count),
            "metrics": _compute_metrics(budget_results),
            "selection_file": selection_file,
        })
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"name": target.name, "path": target.path, "split": target.split, "limit": target.limit},
        "backend": backend.identity(),
        "batch_size": config.batch_size,
        "oracle_policy": {
            "allocation": "deferrals_first_then_lowest_confidence",
            "tie_breaker": "dataset_order",
            "budget_ratio": config.oracle_budget_ratio,
            "budget_count": config.oracle_budget_count,
        },
        "n_rows": len(rows),
        "status_counts": status_counts,
        "metrics": metrics,
        "budget_sweep": budget_sweep,
        "selection_file": "selections.jsonl",
    }
    (output_dir / "selector.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
