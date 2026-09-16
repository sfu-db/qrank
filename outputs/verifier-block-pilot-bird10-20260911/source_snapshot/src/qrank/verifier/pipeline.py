"""Cluster, verify, validate, and annotate SQL and PrepBench datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qrank.llms import LLMBackend

from .cache import SQLiteCache
from .context import build_context, detect_task_type, load_database_schemas
from .models import ClusterInput, QuestionInput, VerificationError, VerificationResult, stable_json
from .prompts import PERSPECTIVES, PROMPT_VERSION
from .workflow import verify_one, verify_many

def question_to_canonical_dict(q: QuestionInput) -> dict[str, Any]:
    return {
        "task_type": q.task_type,
        "question_id": q.question_id,
        "question": q.question,
        "evidence": q.evidence,
        "schema": q.schema,
        "sql_dialect": q.sql_dialect,
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



def result_cluster_dict(c: ClusterInput, vr: VerificationResult, backend: LLMBackend) -> dict[str, Any]:
    return {
        "cluster_id": c.cluster_id,
        "vote_count": c.vote_count,
        "representative_sql": c.representative_sql,
        "labels": vr.labels,
        "reasons": vr.reasons,
        "block_results": vr.block_results,
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



def collect_questions(questions, backend, cache, batch_size, *, perspectives=PERSPECTIVES, audit_path=None):
    items = [(q, c) for q in questions for c in q.clusters]
    results = iter(verify_many(items, backend, cache, batch_size, perspectives=perspectives, audit_path=audit_path))
    rows = []
    for q in questions:
        row = question_to_canonical_dict(q)
        clusters = [result_cluster_dict(c, next(results), backend) for c in q.clusters]
        row["clusters"] = clusters
        row["collection_status"] = (
            "all-invalid-or-no-valid-cluster" if not clusters else
            "ok"
        )
        rows.append(row)
    return rows



def summarize_rows(rows: Sequence[Mapping[str, Any]], perspectives=PERSPECTIVES) -> dict[str, Any]:
    """Count completed judgments; errors have already stopped the workflow."""
    counts = {p: {label: 0 for label in ("S", "U", "C", "NA")} for p in perspectives}
    clusters = [c for q in rows for c in q["clusters"]]
    for c in clusters:
        for p in perspectives:
            counts[p][c["labels"][p]] += 1
    block_counts = {p: {label: 0 for label in ("S", "U", "C", "NA")} for p in perspectives}
    for c in clusters:
        for block in c["block_results"]:
            block_counts[block["perspective"]][block["label"]] += 1
    return {
        "block_label_counts": block_counts,
        "n_questions": len(rows), "n_clusters": len(clusters),
        "n_agreement_clusters": sum(len({c["metadata"]["agreement_cluster_id"]
                                         for c in q["clusters"]}) for q in rows),
        "n_unverified_clusters": 0, "n_failed_verifications": 0,
        "n_all_invalid_or_no_valid_cluster": sum(not q["clusters"] for q in rows),
        "label_counts": counts, "errors": [],
    }


def dataset_questions(args: argparse.Namespace):
    from datasets import Dataset, DatasetDict, load_from_disk

    if Path(args.input).is_file() and Path(args.input).suffix == ".jsonl":
        with Path(args.input).open(encoding="utf-8") as stream:
            dataset = DatasetDict({args.split: Dataset.from_list(
                [json.loads(line) for line in stream if line.strip()])})
    else:
        dataset = load_from_disk(args.input)
    split = dataset[args.split]
    if args.limit is not None:
        if args.limit <= 0:
            raise VerificationError("--limit must be positive")
        split = split.select(range(min(args.limit, len(split))))
    task_type = getattr(args, "task_type", "auto")
    kinds = [detect_task_type(row, task_type) for row in split]
    schemas = load_database_schemas(args) if "text-to-sql" in kinds else {}
    dialect = getattr(args, "sql_dialect", None)
    if dialect is None and schemas and (getattr(args, "db_root", None) or not args.schemas):
        # Automatic discovery under dataset/base and --db-root load BIRD/SQLite.
        dialect = "SQLite"
    questions = []
    question_ids = set()
    for row in split:
        qid = row["question_id"]
        if qid is None or not str(qid).strip() or str(qid) in question_ids:
            raise VerificationError(f"Missing or duplicate question_id: {qid!r}")
        question_ids.add(str(qid))
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
            distinct = {}
            for member in sorted(members):
                distinct.setdefault(candidates[member]["text"].strip(), []).append(member)
            for sql, indices in distinct.items():
                clusters.append(ClusterInput(str(indices[0]), sql, len(indices), {
                    "candidate_indices": indices, "agreement_cluster_id": str(seed),
                }))
        context = build_context(row, task_type=task_type, schemas=schemas,
                                prepbench_root=getattr(args, "prepbench_root", None),
                                sql_dialect=dialect)
        questions.append(QuestionInput(
            str(row["question_id"]), context.question, context.evidence,
            context.context, context.db_id, context.difficulty, tuple(clusters), {},
            context.task_type, context.sql_dialect,
        ))
    return dataset, split, questions



def attach_results(dataset, split, questions, rows, args: argparse.Namespace) -> None:
    from datasets import Features, List, Value

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    annotations = []
    for source, q, result in zip(split, questions, rows, strict=True):
        aligned = [{"cluster_id": "", "status": "invalid", "cache_key": "",
                    "labels": {p: None for p in perspectives},
                    "agreement_cluster_id": "", "block_results": [],
                    "reasons": {p: "invalid or empty candidate" for p in perspectives}}
                   for _ in source["candidates"]]
        for c, r in zip(q.clusters, result["clusters"], strict=True):
            for i in c.metadata["candidate_indices"]:
                aligned[i] = {"cluster_id": c.cluster_id, "status": r["verification"]["status"],
                              "cache_key": r["verification"]["cache_key"],
                              "labels": r["labels"], "reasons": r["reasons"],
                              "block_results": [{**b, "sections": stable_json(b["sections"])}
                                                for b in r["block_results"]],
                              "agreement_cluster_id": c.metadata["agreement_cluster_id"]}
        annotations.append(aligned)
    # Explicit types also handle all-invalid pilot subsets without null-typed fields.
    block_type = List({key: Value("string") for key in (
        "block_id", "scope_id", "perspective", "parent_scope_id", "sections",
        "label", "reason", "cache_key", "status")})
    annotation_type = List({
        "block_results": block_type,
        "cluster_id": Value("string"), "status": Value("string"), "cache_key": Value("string"),
        "labels": {p: Value("string") for p in perspectives},
        "agreement_cluster_id": Value("string"),
        "reasons": {p: Value("string") for p in perspectives},
    })
    if "semantic_verifier" in split.column_names:
        split = split.remove_columns("semantic_verifier")
    features = Features({**split.features, "semantic_verifier": annotation_type})
    dataset[args.split] = split.add_column("semantic_verifier", annotations).cast(features)
    dataset.save_to_disk(args.output)
