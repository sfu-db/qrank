"""Cluster, verify, validate, and annotate SQL and PrepBench datasets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from qrank.llms import BackendError, LLMBackend

from .cache import SQLiteCache
from .context import build_context, detect_task_type, load_database_schemas
from .models import (ClusterInput, QuestionInput, VerificationError, VerificationResult,
                     sha256_text, stable_json)
from .prompts import (PERSPECTIVES, PROMPT_VERSION, SYSTEM_PROMPT, VERDICTS,
                      verifier_user_prompt)

def question_to_canonical_dict(q: QuestionInput) -> dict[str, Any]:
    return {
        "task_type": q.task_type,
        "question_id": q.question_id,
        "question": q.question,
        "evidence": q.evidence,
        "schema": q.schema,
        "db_id": q.db_id,
        "difficulty": q.difficulty,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "vote_count": c.vote_count,
                "representative_sql": c.representative_sql,
                **({"metadata": c.metadata} if c.metadata else {}),
            }
            for c in q.clusters
        ],
        **({"metadata": q.metadata} if q.metadata else {}),
    }



def extract_json_object(text: str) -> dict[str, Any]:
    # Accept surrounding prose or code fences; let the JSON decoder find the end.
    start = text.find("{")
    if start < 0:
        raise VerificationError("Verifier response contains no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"Malformed JSON object: {exc}") from exc
    return obj



def normalize_label(value: Any, field: str) -> tuple[str, str]:
    reason = ""
    if isinstance(value, Mapping):
        label = str(value.get("label", "")).strip().upper()
        reason = str(value.get("reason", "") or "").strip()
    else:
        label = str(value).strip().upper()
    aliases = {
        "SATISFIED": "S", "PASS": "S", "YES": "S",
        "UNRESOLVED": "U", "UNCERTAIN": "U", "UNKNOWN": "U",
        "CONTRADICTED": "C", "CONTRADICTION": "C", "FAIL": "C", "NO": "C",
        "NOT_APPLICABLE": "NA", "N/A": "NA", "NOT APPLICABLE": "NA",
    }
    label = aliases.get(label, label)
    if label not in VERDICTS:
        raise VerificationError(f"{field}: invalid label {label!r}")
    if field == "overall" and label == "NA":
        raise VerificationError("overall may not be NA")
    return label, reason



def parse_verifier_response(text: str) -> tuple[dict[str, str], dict[str, str]]:
    obj = extract_json_object(text)
    labels: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for field in PERSPECTIVES:
        if field not in obj:
            raise VerificationError(f"missing field {field}")
        label, reason = normalize_label(obj[field], field)
        labels[field] = label
        reasons[field] = reason
    return labels, reasons



def make_cache_key(q: QuestionInput, c: ClusterInput, backend: LLMBackend) -> str:
    # Excludes vote count, cluster id, sources, and other ensemble information by design.
    payload = {
        "prompt_version": PROMPT_VERSION,
        "backend_identity": backend.identity(),
        "task_type": q.task_type,
        "system": SYSTEM_PROMPT,
        "question": q.question,
        "evidence": q.evidence,
        "schema": q.schema,
        "sql": c.representative_sql,
    }
    return sha256_text(stable_json(payload))



def parsed_payload(labels: Mapping[str, str], reasons: Mapping[str, str]) -> dict[str, Any]:
    return {field: {"label": labels[field], "reason": reasons.get(field, "")} for field in PERSPECTIVES}



def verify_one(
    q: QuestionInput,
    c: ClusterInput,
    backend: LLMBackend,
    cache: SQLiteCache | None = None,
    retries: int = 2,
    retry_delay: float = 1.0,
) -> VerificationResult:
    return verify_many([(q, c)], backend, cache, retries, retry_delay, 1)[0]



def verify_many(
    items: Sequence[tuple[QuestionInput, ClusterInput]],
    backend: LLMBackend,
    cache: SQLiteCache | None = None,
    retries: int = 2,
    retry_delay: float = 1.0,
    batch_size: int = 8,
) -> list[VerificationResult]:
    if batch_size < 1:
        raise VerificationError("batch_size must be positive")
    results = {}
    pending = []
    keys = [make_cache_key(q, c, backend) for q, c in items]
    for i, key in enumerate(keys):
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            pending.append(i)
        else:
            raw, parsed = cached
            labels = {p: parsed[p]["label"] for p in PERSPECTIVES}
            reasons = {p: parsed[p]["reason"] for p in PERSPECTIVES}
            results[i] = VerificationResult(labels, reasons, raw, "cached", key, 0)

    for attempt in range(1, retries + 2):
        retry = []
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            requests = [{"question_id": items[i][0].question_id,
                         "cluster_id": items[i][1].cluster_id,
                         "system": SYSTEM_PROMPT,
                         "user": verifier_user_prompt(*items[i])} for i in indices]
            try:
                responses = backend.generate_batch(requests)
            except (OSError, BackendError, VerificationError, ValueError) as exc:
                responses = [exc] * len(indices)
            # GPU/runtime and programming errors propagate; retrying the same batch
            # cannot fix an out-of-memory error. Resume with a smaller --batch-size.
            for i, raw in zip(indices, responses, strict=True):
                try:
                    if isinstance(raw, Exception):
                        raise raw
                    labels, reasons = parse_verifier_response(raw)
                except (OSError, BackendError, VerificationError, ValueError) as exc:
                    retry.append(i)
                    labels = {p: "NA" for p in PERSPECTIVES}
                    reasons = {p: f"verification unavailable: {type(exc).__name__}: {exc}"
                               for p in PERSPECTIVES}
                    results[i] = VerificationResult(labels, reasons, "", "failed", keys[i], attempt)
                    continue
                if cache is not None:
                    cache.put(keys[i], raw, parsed_payload(labels, reasons), backend)
                results[i] = VerificationResult(labels, reasons, raw, "ok", keys[i], attempt)
        pending = retry
        if not pending:
            break
        if attempt <= retries:
            time.sleep(retry_delay * (2 ** (attempt - 1)))
    return [results[i] for i in range(len(items))]



def result_cluster_dict(c: ClusterInput, vr: VerificationResult, backend: LLMBackend) -> dict[str, Any]:
    return {
        "cluster_id": c.cluster_id,
        "vote_count": c.vote_count,
        "representative_sql": c.representative_sql,
        "labels": vr.labels,
        "reasons": vr.reasons,
        "reason": vr.reasons.get("overall", ""),
        "verification": {
            "status": vr.status,
            "cache_key": vr.cache_key,
            "attempts": vr.attempts,
            "prompt_version": PROMPT_VERSION,
            "backend": backend.backend_name,
            "model": backend.model,
        },
        **({"metadata": c.metadata} if c.metadata else {}),
    }



def collect_questions(questions, backend, cache, retries, retry_delay, batch_size):
    items = [(q, c) for q in questions for c in q.clusters]
    results = iter(verify_many(items, backend, cache, retries, retry_delay, batch_size))
    rows = []
    for q in questions:
        row = question_to_canonical_dict(q)
        clusters = [result_cluster_dict(c, next(results), backend) for c in q.clusters]
        row["clusters"] = clusters
        row["collection_status"] = (
            "all-invalid-or-no-valid-cluster" if not clusters else
            "partial" if any(c["verification"]["status"] == "failed" for c in clusters) else "ok"
        )
        rows.append(row)
    return rows



def label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {field: {v: 0 for v in ("S", "U", "C", "NA")} for field in PERSPECTIVES}
    for q in rows:
        for c in q.get("clusters", []):
            labels = c.get("labels", {})
            for field in PERSPECTIVES:
                # Prepared inputs have no observations yet; missing is not a verdict.
                v = labels.get(field)
                if v in counts[field]:
                    counts[field][v] += 1
    return counts



def validate_rows(rows: Sequence[Mapping[str, Any]], allow_failed_overall_na: bool = True) -> dict[str, Any]:
    qids: set[str] = set()
    keys: set[tuple[str, str]] = set()
    errors: list[str] = []
    all_invalid = 0
    n_clusters = 0
    unverified = 0
    failed = 0
    for qi, q in enumerate(rows):
        qid = str(q.get("question_id", ""))
        if not qid:
            errors.append(f"row {qi}: missing question_id")
            continue
        if qid in qids:
            errors.append(f"duplicate question_id={qid}")
        qids.add(qid)
        clusters = q.get("clusters", [])
        if not isinstance(clusters, list):
            errors.append(f"question_id={qid}: clusters must be list")
            continue
        if not clusters:
            all_invalid += 1
        for c in clusters:
            n_clusters += 1
            cid = str(c.get("cluster_id", ""))
            key = (qid, cid)
            if key in keys:
                errors.append(f"duplicate cluster key={key}")
            keys.add(key)
            if not str(c.get("representative_sql", "")).strip():
                errors.append(f"cluster key={key}: empty representative_sql")
            try:
                votes = int(c.get("vote_count", 0))
                if votes <= 0:
                    errors.append(f"cluster key={key}: vote_count must be >0")
            except (TypeError, ValueError, OverflowError):
                errors.append(f"cluster key={key}: invalid vote_count")
            status = c.get("verification", {}).get("status", "") if isinstance(c.get("verification"), Mapping) else ""
            if status == "failed":
                failed += 1
            labels = c.get("labels")
            if labels is None:
                # Canonical verifier INPUT is allowed to omit labels.
                unverified += 1
                continue
            if not isinstance(labels, Mapping):
                errors.append(f"cluster key={key}: labels must be object")
                continue
            for field in PERSPECTIVES:
                v = str(labels.get(field, ""))
                if v not in VERDICTS:
                    errors.append(f"cluster key={key}: {field} invalid label={v!r}")
                if field == "overall" and v == "NA" and not (allow_failed_overall_na and status == "failed"):
                    errors.append(f"cluster key={key}: overall cannot be NA unless verification failed")
    return {
        "n_questions": len(rows),
        "n_clusters": n_clusters,
        "n_unverified_clusters": unverified,
        "n_all_invalid_or_no_valid_cluster": all_invalid,
        "n_failed_verifications": failed,
        "label_counts": label_counts(rows),
        "errors": errors,
    }



def dataset_questions(args: argparse.Namespace):
    from datasets import Dataset, DatasetDict, load_from_disk

    try:
        if Path(args.input).is_file() and Path(args.input).suffix == ".jsonl":
            with Path(args.input).open(encoding="utf-8") as stream:
                dataset = DatasetDict({args.split: Dataset.from_list(
                    [json.loads(line) for line in stream if line.strip()])})
        else:
            dataset = load_from_disk(args.input)
    except FileNotFoundError as exc:
        raise VerificationError(
            f"Could not load dataset from {args.input!r}: {exc}. "
            f"Working directory: {Path.cwd()}. "
            f"Resolved input path: {Path(args.input).resolve()}. "
            "Run from the project root or pass --input with an absolute path "
            "to a saved Hugging Face DatasetDict."
        ) from exc
    split = dataset[args.split]
    if args.limit is not None:
        if args.limit <= 0:
            raise VerificationError("--limit must be positive")
        split = split.select(range(min(args.limit, len(split))))
    task_type = getattr(args, "task_type", "auto")
    try:
        kinds = [detect_task_type(row, task_type) for row in split]
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    schemas = load_database_schemas(args) if "text-to-sql" in kinds else {}
    questions = []
    for row in split:
        candidates = row["candidates"]
        n = len(candidates)
        agreement = row.get("agreement")
        if agreement is None:
            # Raw PrepBench exports supply equivalence IDs instead of a matrix.
            if detect_task_type(row, task_type) != "prepbench":
                raise VerificationError("Text-to-SQL rows require an agreement matrix")
            agreement = [[int(i == j or (c.get("cluster_id") is not None and
                          c.get("cluster_id") != "" and c.get("cluster_id") == d.get("cluster_id")))
                          for j, d in enumerate(candidates)] for i, c in enumerate(candidates)]
        if len(agreement) != n or any(len(a) != n for a in agreement):
            raise VerificationError(f"question {row['question_id']}: agreement shape mismatch")
        valid = [i for i, c in enumerate(candidates)
                 if not c.get("invalid_mask", False) and c["text"].strip()]
        remaining = set(valid)
        clusters = []
        while remaining:
            seed = min(remaining)
            members = {seed}
            pending = [seed]
            while pending:
                i = pending.pop()
                for j in valid:
                    if bool(agreement[i][j]) != bool(agreement[j][i]):
                        raise VerificationError(f"question {row['question_id']}: asymmetric agreement")
                    if agreement[i][j] and j not in members:
                        members.add(j)
                        pending.append(j)
            # Do not silently turn non-transitive agreement into equivalence.
            if any(not agreement[i][j] for i in members for j in members if i != j):
                raise VerificationError(f"question {row['question_id']}: non-transitive agreement")
            remaining -= members
            clusters.append(ClusterInput(str(seed), candidates[seed]["text"].strip(),
                                         len(members), {"candidate_indices": sorted(members)}))
        try:
            context = build_context(row, task_type=task_type, schemas=schemas,
                                    prepbench_root=getattr(args, "prepbench_root", None))
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
        questions.append(QuestionInput(
            str(row["question_id"]), context.question, context.evidence,
            context.context, context.db_id, context.difficulty, tuple(clusters), {},
            context.task_type,
        ))
    report = validate_rows([question_to_canonical_dict(q) for q in questions])
    if report["errors"]:
        raise VerificationError("\n".join(report["errors"][:30]))
    return dataset, split, questions



def attach_results(dataset, split, questions, rows, args: argparse.Namespace) -> None:
    from datasets import Features, List, Value

    annotations = []
    for source, q, result in zip(split, questions, rows, strict=True):
        aligned = [{"cluster_id": "", "status": "invalid", "cache_key": "",
                    "labels": {p: "NA" for p in PERSPECTIVES},
                    "reasons": {p: "invalid or empty candidate" for p in PERSPECTIVES}}
                   for _ in source["candidates"]]
        for c, r in zip(q.clusters, result["clusters"], strict=True):
            for i in c.metadata["candidate_indices"]:
                aligned[i] = {"cluster_id": c.cluster_id, "status": r["verification"]["status"],
                              "cache_key": r["verification"]["cache_key"],
                              "labels": r["labels"], "reasons": r["reasons"]}
        annotations.append(aligned)
    # Explicit types also handle all-invalid pilot subsets without null-typed fields.
    annotation_type = List({
        "cluster_id": Value("string"), "status": Value("string"), "cache_key": Value("string"),
        "labels": {p: Value("string") for p in PERSPECTIVES},
        "reasons": {p: Value("string") for p in PERSPECTIVES},
    })
    if "semantic_verifier" in split.column_names:
        split = split.remove_columns("semantic_verifier")
    features = Features({**split.features, "semantic_verifier": annotation_type})
    dataset[args.split] = split.add_column("semantic_verifier", annotations).cast(features)
    dataset.save_to_disk(args.output)
