"""Independent perspective verification with deterministic SQL decomposition."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Sequence

from qrank.llms import LLMBackend

from .cache import SQLiteCache
from .models import ClusterInput, QuestionInput, VerificationResult, sha256_text, stable_json
from .prompts import PROMPT_VERSION, system_prompt, verifier_user_prompt, block_prompt
from .responses import PERSPECTIVES, response_schema, parse_response
from .sql import decompose_sql, format_sql_blocks


def audit_event(path: str | Path | None, **event: Any) -> None:
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(stable_json({"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **event}) + "\n")


def request_key(request: dict[str, Any], backend: LLMBackend) -> str:
    return sha256_text(stable_json({
        "prompt_version": PROMPT_VERSION, "backend_identity": backend.identity(),
        "system": request["system"], "user": request["user"], "response_schema": request["response_schema"],
    }))


def verify_one(q: QuestionInput, c: ClusterInput, backend: LLMBackend,
               cache: SQLiteCache | None = None, *, perspectives: Sequence[str] = PERSPECTIVES,
               audit_path: str | Path | None = None) -> VerificationResult:
    return verify_many([(q, c)], backend, cache, batch_size=1,
                       perspectives=perspectives, audit_path=audit_path)[0]


def verify_many(items: Sequence[tuple[QuestionInput, ClusterInput]], backend: LLMBackend,
                cache: SQLiteCache | None = None, batch_size: int = 8, *,
                perspectives: Sequence[str] = PERSPECTIVES,
                audit_path: str | Path | None = None) -> list[VerificationResult]:
    if not perspectives or len(set(perspectives)) != len(perspectives) or not set(perspectives) <= set(PERSPECTIVES):
        raise ValueError(f"Choose distinct perspectives from {PERSPECTIVES}")
    results, pending, candidate_keys = {}, {}, []
    summaries = {}
    for q, c in items:
        audit_event(audit_path, stage="prepare", question_id=q.question_id, cluster_id=c.cluster_id,
                    candidate=c.representative_sql, dialect=q.sql_dialect)
        summary = ""
        if q.task_type == "text-to-sql":
            ast_key = (c.representative_sql, q.sql_dialect)
            if ast_key not in summaries:
                extracted = decompose_sql(*ast_key)
                summaries[ast_key] = (format_sql_blocks(extracted, q.sql_dialect), extracted)
            summary, blocks = summaries[ast_key]
        else:
            blocks = [{"scope_id": "code", "parent_scope_id": "root",
                       "kind": "PYTHON", "sections": {}}]
        user = verifier_user_prompt(q, c, summary)
        keys = {}
        for block in blocks:
            for perspective in perspectives:
                request = {"question_id": q.question_id, "cluster_id": c.cluster_id,
                           "system": system_prompt(q, perspective),
                           "user": user,
                           "response_schema": response_schema(perspective)}
                if q.task_type == "text-to-sql":
                    request["user"] += "\n\n" + block_prompt(block, perspective)
                key = request_key(request, backend)
                keys[(block["scope_id"], perspective)] = (key, block)
                if key in results or key in pending:
                    continue
                cached = cache.get(key) if cache is not None else None
                if cached is None:
                    pending[key] = (request, perspective)
                else:
                    raw, parsed = cached
                    results[key] = (parsed["label"], parsed["reason"], raw, "cached")
        candidate_keys.append(keys)

    pending_keys = list(pending)
    for start in range(0, len(pending_keys), batch_size):
        batch = pending_keys[start:start + batch_size]
        requests = [pending[key][0] for key in batch]
        audit_event(audit_path, stage="perspective", status="requested", requests=requests)
        responses = backend.generate_batch(requests)
        # Preserve every raw completion before interpreting the first one.
        for key, request, raw in zip(batch, requests, responses, strict=True):
            audit_event(audit_path, stage="perspective", status="generated", perspective=pending[key][1],
                        request=request, raw_text=raw)
        for key, raw in zip(batch, responses, strict=True):
            label, reason = parse_response(raw, pending[key][1])
            if cache is not None:
                cache.put(key, raw, {"label": label, "reason": reason}, backend)
            results[key] = (label, reason, raw, "ok")

    output = []
    for keys in candidate_keys:
        status = "cached" if all(results[k][3] == "cached" for k, _ in keys.values()) else "ok"
        judgments = [
            {"block_id": scope + ":" + p, "scope_id": scope, "perspective": p,
             "parent_scope_id": block["parent_scope_id"],
             "sections": block["sections"],
             "label": results[k][0], "reason": results[k][1], "cache_key": k,
             "status": results[k][3]}
            for (scope, p), (k, block) in keys.items()
        ]
        # Compatibility fields expose root-scope signals, never an aggregation.
        root = [j for j in judgments if j["parent_scope_id"] == "root"]
        output.append(VerificationResult(
            {j["perspective"]: j["label"] for j in root},
            {j["perspective"]: j["reason"] for j in root},
            stable_json({scope + ":" + p: results[k][2] for (scope, p), (k, _) in keys.items()}),
            status, sha256_text(stable_json({j["block_id"]: j["cache_key"] for j in judgments})),
            int(status == "ok"), judgments,
        ))
    return output
