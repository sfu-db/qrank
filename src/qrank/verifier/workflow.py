"""Grounded local SQL signals with shared candidate-independent requirements."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Sequence

from qrank.llms import LLMBackend

from .cache import SQLiteCache
from .models import ClusterInput, QuestionInput, VerificationResult, sha256_text, stable_json
from .prompts import (
    BLOCK_SECTIONS, PROMPT_VERSION, REQUIREMENTS_VERSION, grounded_system, grounded_user,
    requirement_source_inputs, requirements_request, system_prompt, verifier_user_prompt,
)
from .responses import (
    PERSPECTIVES, grounded_response_schema, parse_grounded, parse_requirements, parse_response, response_schema,
)
from .sql import decompose_sql, format_sql_blocks


def audit_event(path: str | Path | None, **event: Any) -> None:
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(stable_json({"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **event}) + "\n")


def request_key(request: dict[str, Any], backend: LLMBackend, *, version: str = PROMPT_VERSION) -> str:
    return sha256_text(stable_json({
        "prompt_version": version, "backend_identity": backend.identity(),
        "system": request["system"], "user": request["user"], "response_schema": request["response_schema"],
    }))


def complete_requests(requests: dict[str, tuple[dict, Callable]], backend: LLMBackend,
                      cache: SQLiteCache | None, batch_size: int, audit_path, stage: str) -> dict:
    """Deduplicate, validate cached/raw responses, and preserve a batch before parsing."""
    results, pending = {}, []
    for key, (request, parse) in requests.items():
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            pending.append(key)
        else:
            raw, _ = cached
            parsed = parse(raw)
            results[key] = (parsed, raw, "cached")
            audit_event(audit_path, stage=stage, status="cached", cache_key=key, request=request)
    for start in range(0, len(pending), batch_size):
        keys = pending[start:start + batch_size]
        batch = [requests[key][0] for key in keys]
        audit_event(audit_path, stage=stage, status="requested", requests=batch)
        responses = backend.generate_batch(batch)
        for key, request, raw in zip(keys, batch, responses, strict=True):
            audit_event(audit_path, stage=stage, status="generated", cache_key=key,
                        request=request, raw_text=raw)
        for key, raw in zip(keys, responses, strict=True):
            parsed = requests[key][1](raw)
            if cache is not None:
                cache.put(key, raw, parsed, backend,
                          prompt_version=REQUIREMENTS_VERSION if stage == "requirements" else PROMPT_VERSION)
            results[key] = (parsed, raw, "ok")
    return results


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
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    plans, summaries, extractions = [], {}, {}
    for q, c in items:
        audit_event(audit_path, stage="prepare", question_id=q.question_id, cluster_id=c.cluster_id,
                    candidate=c.representative_sql, dialect=q.sql_dialect)
        summary, contract_key = "", ""
        if q.task_type == "text-to-sql":
            ast_key = (c.representative_sql, q.sql_dialect)
            if ast_key not in summaries:
                blocks = decompose_sql(*ast_key)
                summaries[ast_key] = (format_sql_blocks(blocks, q.sql_dialect), blocks)
            summary, blocks = summaries[ast_key]
            request = requirements_request(q)
            contract_key = request_key(request, backend, version=REQUIREMENTS_VERSION)
            extractions.setdefault(contract_key, (request, lambda raw, q=q: parse_requirements(raw, q)))
        else:
            blocks = [{"scope_id": "code", "parent_scope_id": "root", "kind": "PYTHON",
                       "sections": {}, "reachable": True, "dependency_scope_ids": []}]
        plans.append((q, c, summary, blocks, contract_key))
    contracts = complete_requests(extractions, backend, cache, batch_size, audit_path, "requirements")

    checks, candidate_checks = {}, []
    for q, c, summary, blocks, contract_key in plans:
        contract = contracts[contract_key][0] if contract_key else {}
        targets = []
        for block in blocks:
            if not block["reachable"]:
                continue
            for perspective in perspectives:
                sections = {heading: block["sections"][heading] for heading in BLOCK_SECTIONS[perspective]} if contract_key else {}
                assigned = [r["requirement_id"] for r in contract.get("requirements", [])
                            if r["dimension"] == perspective]
                # Candidate-introduced operations can harm output coverage/grain
                # even when the question has no explicit ranking/relation request.
                guard_sections = {
                    "constraint_semantics": ("FILTERS (WHERE)", "HAVING", "QUALIFY"),
                    "ranking_semantics": ("LIMIT", "OFFSET"),
                    "relation_semantics": ("JOINS", "GROUP BY", "DISTINCT", "SET OPERATION", "WINDOW EXPRESSIONS"),
                }.get(perspective, ())
                if not assigned and any(sections.get(name) for name in guard_sections):
                    assigned = [r["requirement_id"] for r in contract.get("requirements", [])
                                if r["dimension"] == "output_semantics"]
                targets.append({
                    "block_id": block["scope_id"] + ":" + perspective,
                    "scope_id": block["scope_id"], "parent_scope_id": block["parent_scope_id"],
                    "scope_role": "root_result" if block["parent_scope_id"] == "root" else "intermediate",
                    "perspective": perspective, "sections": sections,
                    "dependency_scope_ids": block["dependency_scope_ids"],
                    "observation_kind": "local_contribution", "requirement_id": "",
                    "allowed_requirement_ids": assigned,
                })
        if contract_key:
            root = next(b for b in blocks if b["parent_scope_id"] == "root")
            for requirement in contract["requirements"]:
                if requirement["dimension"] not in perspectives:
                    continue
                targets.append({
                    "block_id": "coverage:" + requirement["requirement_id"],
                    "scope_id": root["scope_id"], "parent_scope_id": "root",
                    "scope_role": "requirement_coverage", "perspective": requirement["dimension"],
                    "sections": root["sections"], "dependency_scope_ids": root["dependency_scope_ids"],
                    "observation_kind": "requirement_coverage", "requirement_id": requirement["requirement_id"],
                    "allowed_requirement_ids": [requirement["requirement_id"]],
                })
        keys = []
        for target in targets:
            perspective = target["perspective"]
            if contract_key:
                schema = grounded_response_schema(target, blocks)
                request = {"question_id": q.question_id, "cluster_id": c.cluster_id,
                           "system": grounded_system(perspective, schema),
                           "user": grounded_user(q, c, summary, contract, target),
                           "response_schema": schema}
                parse = lambda raw, q=q, c=c, contract=contract, blocks=blocks, target=target: parse_grounded(
                    raw, q=q, sql=c.representative_sql, requirements=contract, blocks=blocks, target=target)
            else:
                request = {"question_id": q.question_id, "cluster_id": c.cluster_id,
                           "system": system_prompt(q, perspective),
                           "user": verifier_user_prompt(q, c), "response_schema": response_schema(perspective)}
                def parse(raw, p=perspective):
                    label, reason = parse_response(raw, p)
                    return {"label": label, "reason": reason}
            key = request_key(request, backend)
            checks.setdefault(key, (request, parse))
            keys.append((key, target))
        candidate_checks.append(keys)
    results = complete_requests(checks, backend, cache, batch_size, audit_path, "checks")

    output = []
    for (q, c, _, _, contract_key), keys in zip(plans, candidate_checks, strict=True):
        statuses = [results[key][2] for key, _ in keys]
        if contract_key:
            statuses.append(contracts[contract_key][2])
        status = "cached" if all(s == "cached" for s in statuses) else "ok"
        observations = []
        for key, target in keys:
            parsed, _, check_status = results[key]
            observations.append({
                **target,
                "requirement_ids": [], "source_refs": [], "sql_evidence": [],
                "inspected_scope_ids": [], "missing_behavior": None, "conflict_type": None,
                "unresolved_reason": None, "applicability_reason": None,
                **parsed,
                "cache_key": key, "call_id": key, "status": check_status,
                "contract_key": contract_key,
                "source_id": sha256_text(stable_json({"backend": backend.identity(),
                    "prompt_version": PROMPT_VERSION, "observation_kind": target["observation_kind"],
                    "perspective": target["perspective"]})),
                "dependency_group_ids": [sha256_text(stable_json([contract_key, c.representative_sql, rid]))
                                         for rid in parsed.get("requirement_ids", [])],
            })
        judgments = [j for j in observations if j["observation_kind"] == "local_contribution"]
        coverage = [j for j in observations if j["observation_kind"] == "requirement_coverage"]
        root = [j for j in judgments if j["parent_scope_id"] == "root"]
        contract_record = ({**contracts[contract_key][0], "cache_key": contract_key,
                            "prompt_version": REQUIREMENTS_VERSION,
                            "extractor": backend.identity(),
                            "source_inputs": requirement_source_inputs(q)} if contract_key else {})
        output.append(VerificationResult(
            labels={j["perspective"]: j["label"] for j in root},
            reasons={j["perspective"]: j["reason"] for j in root},
            raw_text=stable_json({target["block_id"]: results[key][1] for key, target in keys}),
            status=status, cache_key=sha256_text(stable_json({
                "contract": contract_key, "checks": {j["block_id"]: j["cache_key"] for j in observations}})),
            attempts=int(status == "ok"), block_results=judgments, requirement_results=coverage,
            requirements=contract_record,
        ))
    return output
