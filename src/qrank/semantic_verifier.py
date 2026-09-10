#!/usr/bin/env python3
"""Annotate a saved Hugging Face DatasetDict with semantic SQL judgments.

Invalid candidates are filtered before clustering by the agreement matrix.
Each cluster is verified independently using only question, evidence, schema,
and representative SQL. Successful responses are cached in SQLite; failed
calls yield NA labels. Annotations are aligned with the original candidates
and saved alongside the original columns in the output DatasetDict.

    uv run --no-sync qrank-verifier transform-dataset
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Features, List, Value, load_from_disk

PROMPT_VERSION = "qrank-svem-semantic-v1"
PERSPECTIVES = (
    "output_semantics",
    "constraint_semantics",
    "ranking_semantics",
    "relation_semantics",
    "overall",
)
VERDICTS = {"S", "U", "C", "NA"}

SYSTEM_PROMPT = r"""You are a semantic verifier for candidate SQL answers.

Your task is NOT to generate or repair SQL and NOT to compare this candidate to
other candidates. Judge only whether the supplied candidate SQL semantically
satisfies the supplied natural-language question, evidence/hints, and schema or
context when present.

CRITICAL INDEPENDENCE RULES
- Do not infer or use model votes, source identities, majority/minority status,
  ensemble scores, or other candidate answers. They are intentionally hidden.
- Do not assume a syntactically valid or plausible SQL query is semantically
  correct.
- Do not reward the mere presence of SQL syntax such as LIMIT, GROUP BY, AVG,
  COUNT, DISTINCT, or JOIN. Check whether it implements what the question asks.
- Do not penalize a candidate merely for using an alternative but semantically
  equivalent SQL formulation.

VERDICT LABELS
- S = SATISFIED: available information supports that this semantic obligation is
  correctly implemented by the candidate.
- C = CONTRADICTED: there is a concrete semantic mismatch between the task and
  the candidate on this perspective.
- U = UNRESOLVED: the perspective is applicable, but available information is
  insufficient to determine S or C reliably. Use U instead of guessing.
- NA = NOT APPLICABLE: the perspective is not required for this question.

PERSPECTIVES
1. output_semantics
   Does the SQL return the requested entity/value/attribute and, when relevant,
   use the correct aggregation function and aggregation target? Examples of
   contradictions include returning an ID when a count is requested, AVG(age)
   when AVG(salary) is requested, or returning the wrong requested attribute.

2. constraint_semantics
   Does the SQL use the correct population, predicates, attributes, comparison
   operators, literal values, dates, ranges, temporal restrictions, and other
   task conditions? A superficially similar filter on the wrong field is C.

3. ranking_semantics
   If the task requires ranking, min/max, top-k, ordering, or limiting, does the
   SQL rank the correct quantity, in the correct direction, with the correct k?
   If no ranking/extreme/top-k semantics are required, return NA.

4. relation_semantics
   When relational/schema reasoning matters, are tables/entities, joins, join
   keys, grouping entity, relationship direction, and repeated-record/unit-of-
   analysis semantics appropriate? If no meaningful relational choice is needed,
   return NA. If schema/context is insufficient to decide, return U.

5. overall
   Holistic semantic judgment of the candidate. Use C when there is a definite
   essential mismatch anywhere; S when the candidate appears to satisfy the task
   with no material unresolved issue; U when a material issue cannot be resolved.
   Do not use NA for overall.

OUTPUT CONTRACT
Return exactly one JSON object and no prose outside it. Use this schema:
{
  "output_semantics": {"label": "S|U|C|NA", "reason": "short concrete reason"},
  "constraint_semantics": {"label": "S|U|C|NA", "reason": "short concrete reason"},
  "ranking_semantics": {"label": "S|U|C|NA", "reason": "short concrete reason"},
  "relation_semantics": {"label": "S|U|C|NA", "reason": "short concrete reason"},
  "overall": {"label": "S|U|C", "reason": "short concrete reason"}
}

Reasons should identify the task-SQL relationship, not merely restate the label.
"""


class VerificationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ClusterInput:
    cluster_id: str
    representative_sql: str
    vote_count: int
    # Kept for output/audit only; NEVER passed to the verifier prompt.
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class QuestionInput:
    question_id: str
    question: str
    evidence: str
    schema: str
    db_id: str
    difficulty: str
    clusters: tuple[ClusterInput, ...]
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class VerificationResult:
    labels: dict[str, str]
    reasons: dict[str, str]
    raw_text: str
    status: str
    cache_key: str
    attempts: int


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def question_to_canonical_dict(q: QuestionInput) -> dict[str, Any]:
    return {
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


def verifier_user_prompt(q: QuestionInput, c: ClusterInput) -> str:
    # IMPORTANT: no vote_count, cluster_id, source metadata, or competing candidate info.
    parts = [
        "QUESTION:\n" + q.question,
        "EVIDENCE / HINTS:\n" + (q.evidence if q.evidence else "<none provided>"),
        "SCHEMA / CONTEXT:\n" + (q.schema if q.schema else "<none provided>"),
        "CANDIDATE SQL:\n" + c.representative_sql,
    ]
    return "\n\n".join(parts)


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


class VerifierBackend:
    backend_name = "abstract"

    def __init__(self, model: str):
        self.model = model

    def generate(self, *, question_id: str, cluster_id: str, system: str, user: str) -> str:
        raise NotImplementedError

    def generate_batch(self, requests: Sequence[dict[str, str]]) -> list[str]:
        return [self.generate(**request) for request in requests]

    def identity(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "model": self.model}


class TransformersBackend(VerifierBackend):
    """Load one local causal LM and serialize access to its generation state."""

    backend_name = "transformers"

    def __init__(self, model: str, device: str, dtype: str, temperature: float,
                 max_tokens: int, revision: str = "main"):
        super().__init__(model)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if max_tokens <= 0 or temperature < 0:
            raise VerificationError("max-tokens must be positive and temperature nonnegative")
        self.torch = torch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, device_map=device,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        ).eval()
        self.config = {
            "revision": revision, "resolved_revision": getattr(self.lm.config, "_commit_hash", None),
            "device": device, "dtype": str(self.lm.dtype),
            "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": False, "top_p": 0.8, "top_k": 20,
            "transformers_version": transformers.__version__,
        }

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), **self.config}

    def generate(self, *, question_id: str, cluster_id: str, system: str, user: str) -> str:
        return self.generate_batch([{"system": system, "user": user}])[0]

    def generate_batch(self, requests: Sequence[dict[str, str]]) -> list[str]:
        with self.lock, self.torch.inference_mode():
            prompts = [self.tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for r in requests]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(self.lm.device)
            length = inputs["input_ids"].shape[-1]
            if length + self.max_tokens > self.lm.config.max_position_embeddings:
                raise VerificationError("Prompt plus generation exceeds model context window")
            sampling = ({"temperature": self.temperature, "top_p": 0.8, "top_k": 20}
                        if self.temperature > 0 else {})
            output = self.lm.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id, **sampling,
            )
            return [text.strip() for text in self.tokenizer.batch_decode(
                output[:, length:], skip_special_tokens=True)]


class SQLiteCache:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS verifier_cache (
                    cache_key TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    parsed_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                )
            """)
            con.commit()

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=60)
            self._local.con = con
        return con

    def get(self, key: str) -> tuple[str, dict[str, Any]] | None:
        row = self._con().execute(
            "SELECT raw_text, parsed_json FROM verifier_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if not row:
            return None
        return str(row[0]), json.loads(row[1])

    def put(self, key: str, raw_text: str, parsed: dict[str, Any], backend: VerifierBackend) -> None:
        con = self._con()
        con.execute(
            """INSERT OR REPLACE INTO verifier_cache
               (cache_key,raw_text,parsed_json,model,backend,prompt_version,created_utc)
               VALUES (?,?,?,?,?,?,?)""",
            (
                key,
                raw_text,
                stable_json(parsed),
                backend.model,
                backend.backend_name,
                PROMPT_VERSION,
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
        )
        con.commit()


def make_cache_key(q: QuestionInput, c: ClusterInput, backend: VerifierBackend) -> str:
    # Excludes vote count, cluster id, sources, and other ensemble information by design.
    payload = {
        "prompt_version": PROMPT_VERSION,
        "backend_identity": backend.identity(),
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
    backend: VerifierBackend,
    cache: SQLiteCache,
    retries: int,
    retry_delay: float,
) -> VerificationResult:
    return verify_many([(q, c)], backend, cache, retries, retry_delay, 1)[0]


def verify_many(
    items: Sequence[tuple[QuestionInput, ClusterInput]],
    backend: VerifierBackend,
    cache: SQLiteCache,
    retries: int,
    retry_delay: float,
    batch_size: int,
) -> list[VerificationResult]:
    if batch_size < 1:
        raise VerificationError("--batch-size must be positive")
    results = {}
    pending = []
    keys = [make_cache_key(q, c, backend) for q, c in items]
    for i, key in enumerate(keys):
        cached = cache.get(key)
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
            except (OSError, VerificationError, ValueError) as exc:
                responses = [exc] * len(indices)
            # GPU/runtime and programming errors propagate; retrying the same batch
            # cannot fix an out-of-memory error. Resume with a smaller --batch-size.
            for i, raw in zip(indices, responses, strict=True):
                try:
                    if isinstance(raw, Exception):
                        raise raw
                    labels, reasons = parse_verifier_response(raw)
                except (OSError, VerificationError, ValueError) as exc:
                    retry.append(i)
                    labels = {p: "NA" for p in PERSPECTIVES}
                    reasons = {p: f"verification unavailable: {type(exc).__name__}: {exc}"
                               for p in PERSPECTIVES}
                    results[i] = VerificationResult(labels, reasons, "", "failed", keys[i], attempt)
                    continue
                cache.put(keys[i], raw, parsed_payload(labels, reasons), backend)
                results[i] = VerificationResult(labels, reasons, raw, "ok", keys[i], attempt)
        pending = retry
        if not pending:
            break
        if attempt <= retries:
            time.sleep(retry_delay * (2 ** (attempt - 1)))
    return [results[i] for i in range(len(items))]


def result_cluster_dict(c: ClusterInput, vr: VerificationResult, backend: VerifierBackend) -> dict[str, Any]:
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


def make_backend(args: argparse.Namespace) -> VerifierBackend:
    return TransformersBackend(args.model, args.device, args.dtype,
                               args.temperature, args.max_tokens, args.revision)


def load_database_schemas(args: argparse.Namespace) -> dict[str, str]:
    """Read only schema metadata; never execute candidate SQL or read gold SQL."""
    base = Path(__file__).resolve().parents[2] / "dataset/base"
    schema_file = Path(args.schemas) if args.schemas else None
    if schema_file is None and not getattr(args, "db_root", None) and base.exists():
        for filename in ("schema.json", "dev_tables.json"):
            matches = sorted(base.rglob(filename))
            if len(matches) > 1:
                raise VerificationError(f"Multiple {filename} files found; specify --schemas")
            if matches:
                schema_file = matches[0]
                break
    if schema_file:
        tables = json.loads(schema_file.read_text(encoding="utf-8"))
        if isinstance(tables, dict):
            schemas = {}
            for db_id, definitions in tables.items():
                if isinstance(definitions, str):
                    schema = definitions
                elif isinstance(definitions, list) and all(isinstance(s, str) for s in definitions):
                    schema = "\n\n".join(definitions)
                else:
                    raise VerificationError(f"Invalid schema definitions for {db_id!r} in {schema_file}")
                if not schema.strip():
                    raise VerificationError(f"Empty schema for {db_id!r} in {schema_file}")
                schemas[str(db_id)] = schema
            if not schemas:
                raise VerificationError(f"No database schemas found in {schema_file}")
            return schemas
        if not isinstance(tables, list) or not tables or any(
            not isinstance(t, dict) or "db_id" not in t for t in tables
        ):
            raise VerificationError(f"Expected a database-to-DDL mapping or BIRD tables list in {schema_file}")
        schemas = {str(t["db_id"]): stable_json(t) for t in tables}
        if len(schemas) != len(tables):
            raise VerificationError(f"Duplicate database IDs in {schema_file}")
        return schemas
    root = Path(args.db_root) if getattr(args, "db_root", None) else base
    if getattr(args, "db_root", None) and not root.is_dir():
        raise VerificationError(f"Database directory does not exist: {root}")
    schemas = {}
    for path in sorted(root.rglob("*.sqlite")):
        if path.stem in schemas:
            raise VerificationError(f"Duplicate database {path.stem!r}; narrow --db-root")
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            definitions = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
            schemas[path.stem] = "\n\n".join(row[0] for row in definitions)
    if getattr(args, "db_root", None) and not schemas:
        raise VerificationError(f"No .sqlite databases found under {root}")
    return schemas


def dataset_questions(args: argparse.Namespace):
    try:
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
    schemas = load_database_schemas(args)
    questions = []
    for row in split:
        candidates = row["candidates"]
        n = len(candidates)
        agreement = row["agreement"]
        if len(agreement) != n or any(len(a) != n for a in agreement):
            raise VerificationError(f"question {row['question_id']}: agreement shape mismatch")
        valid = [i for i, c in enumerate(candidates)
                 if not c["invalid_mask"] and c["text"].strip()]
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
        metadata = row.get("metadata") or {}
        db_id = str(metadata.get("db_id") or row.get("group_id") or "")
        if schemas and db_id not in schemas:
            raise VerificationError(f"No schema found for database {db_id!r}")
        questions.append(QuestionInput(
            str(row["question_id"]), row["question"], row.get("evidence") or "",
            schemas.get(db_id, row.get("context") or ""), db_id,
            str(metadata.get("difficulty") or ""), tuple(clusters), {},
        ))
    report = validate_rows([question_to_canonical_dict(q) for q in questions])
    if report["errors"]:
        raise VerificationError("\n".join(report["errors"][:30]))
    return dataset, split, questions


def attach_results(dataset, split, questions, rows, args: argparse.Namespace) -> None:
    annotations = []
    for source, q, result in zip(split, questions, rows, strict=True):
        aligned = [{"cluster_id": "", "status": "invalid", "cache_key": "",
                    "labels": {p: "NA" for p in PERSPECTIVES},
                    "reasons": {p: "invalid or empty candidate SQL" for p in PERSPECTIVES}}
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
          f"{sum(not q.schema for q in questions)} questions without schema context.",
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
        "alignment": "same order and length as candidates; invalid candidates have status=invalid",
        **report,
    }, indent=2))
    print(f"Saved transformed DatasetDict to {output}", flush=True)
    return 0


def add_transformers_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--revision", default="main", help="Hugging Face revision or commit")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect frozen QRank semantic-verifier S/U/C measurements")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("transform-dataset", help="annotate saved BIRD data with local Qwen3 via Transformers")
    a.add_argument("--input", default="dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5")
    a.add_argument("--output", default="dataset/processed")
    a.add_argument("--split", default="test")
    a.add_argument("--schemas", help="schema.json or BIRD tables JSON; otherwise discover under dataset/base")
    a.add_argument("--db-root", help="directory containing BIRD .sqlite databases (searched recursively)")
    a.add_argument("--limit", type=int, help="process only the first N rows for a pilot")
    a.add_argument("--model", default="Qwen/Qwen3-1.7B")
    a.add_argument("--cache")
    a.add_argument("--batch-size", type=int, default=8, help="candidate prompts per generation batch")
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--max-tokens", type=int, default=700)
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
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
