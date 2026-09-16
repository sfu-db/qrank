"""Shared verification inputs, results, errors, and serialization helpers."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

class VerificationError(RuntimeError):
    """A verifier response or dataset invariant was violated."""



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
    task_type: str = "text-to-sql"
    sql_dialect: str | None = None


@dataclasses.dataclass(frozen=True)
class RequirementSet:
    """Candidate-independent obligations, with source quotes and ambiguity."""

    requirements: tuple[dict[str, Any], ...]
    unresolved: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"requirements": list(self.requirements), "unresolved": list(self.unresolved)}



@dataclasses.dataclass(frozen=True)
class VerificationResult:
    labels: dict[str, str]
    reasons: dict[str, str]
    raw_text: str
    status: str
    cache_key: str
    attempts: int
    block_results: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    requirement_results: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    requirements: dict[str, Any] = dataclasses.field(default_factory=dict)



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
