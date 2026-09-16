"""Application-facing semantic verification without dataset setup."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qrank.config import VerifierConfig
from qrank.llms import LLMBackend, load_backend

from .cache import SQLiteCache
from .context import build_context
from .models import ClusterInput, QuestionInput, VerificationResult
from .workflow import verify_many
from .responses import PERSPECTIVES


class Verifier:
    """Load the configured backend once and reuse it across requests.

    Supply a config_path or an already-loaded backend. Calls are synchronous;
    applications manage concurrency and optional response caching.
    """

    def __init__(
        self, backend: LLMBackend | None = None, *, config_path: str | Path | None = None,
        cache: SQLiteCache | None = None,
        batch_size: int | None = None, audit_path: str | Path | None = None,
    ):
        if backend is not None and config_path is not None:
            raise ValueError("Supply either backend or config_path, not both")
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        if backend is None:
            if config_path is None:
                raise ValueError("config_path is required when no backend is supplied")
            config = VerifierConfig.from_file(config_path)
            if batch_size is None:
                batch_size = config.batch_size
            backend = load_backend(config.backend, **config.backend_options())
        self.backend = backend
        self.cache = cache
        self.audit_path = audit_path
        self.batch_size = batch_size if batch_size is not None else 8

    def verify(
        self, question: str, candidate: str, *, context: str = "",
        evidence: str = "", task_type: str = "auto", sql_dialect: str | None = None,
        perspective: str | None = None,
    ) -> VerificationResult:
        """Collect grounded SQL block and requirement-coverage signals without execution.

        requirements retains the shared, source-grounded question contract.
        block_results and requirement_results contain separate weak observations;
        labels/reasons expose only root block contributions, never an aggregation.
        PrepBench retains one code-level check per perspective.

        Supply SQL DDL/dialect or table schemas/samples/runtime details in context.
        All SQL parsing, JSON, runtime, and cache errors propagate immediately.
        There are no retries, repaired responses, or failed-result placeholders.
        """
        return self.verify_candidates(
            question, [candidate], context=context, evidence=evidence,
            task_type=task_type, sql_dialect=sql_dialect, perspective=perspective,
        )[0]

    def verify_candidates(
        self, question: str, candidates: Sequence[str], *, context: str = "",
        evidence: str = "", task_type: str = "auto", sql_dialect: str | None = None,
        perspective: str | None = None,
    ) -> list[VerificationResult]:
        """Judge candidates for one task independently, preserving input order.

        Identical candidates share checks; SQL requirements are extracted once per
        distinct question/context within the call and reused across candidates.
        A supplied SQLiteCache also reuses requirements across calls. Distinct
        candidates are checked separately; observations can remain correlated.
        An empty candidate list returns [].
        Auto detection uses the same solve(tables) heuristic as the dataset CLI.
        """
        if isinstance(candidates, (str, bytes)):
            raise ValueError("candidates must be a sequence of strings, not a string")
        if any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError("Every candidate must be a nonempty string")
        task = build_context({
            "question": question, "context": context, "evidence": evidence,
            "candidates": [{"text": c} for c in candidates],
        }, task_type=task_type, sql_dialect=sql_dialect)
        clusters = tuple(ClusterInput(str(i), c, 1, {}) for i, c in enumerate(candidates))
        q = QuestionInput("", task.question, task.evidence, task.context,
                          "", "", clusters, {}, task.task_type, task.sql_dialect)
        return verify_many(
            [(q, c) for c in clusters], self.backend, self.cache,
            self.batch_size, perspectives=(perspective,) if perspective else PERSPECTIVES,
            audit_path=self.audit_path,
        )
