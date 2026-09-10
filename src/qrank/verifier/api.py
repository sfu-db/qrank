"""Application-facing semantic verification without dataset setup."""
from __future__ import annotations

from typing import Sequence

from qrank import config
from qrank.llms import LLMBackend, load_backend

from .cache import SQLiteCache
from .context import build_context
from .models import ClusterInput, QuestionInput, VerificationResult
from .pipeline import verify_many


class Verifier:
    """Load the configured backend once and reuse it across requests.

    An explicit backend overrides qrank.config. Calls are synchronous;
    applications manage concurrency and optional response caching.
    """

    def __init__(
        self, backend: LLMBackend | None = None, *, cache: SQLiteCache | None = None,
        retries: int = 2, retry_delay: float = 1.0, batch_size: int = 8,
    ):
        self.backend = backend if backend is not None else load_backend(
            config.LLM_BACKEND, **config.LLM_OPTIONS,
        )
        self.cache = cache
        self.retries = retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size

    def verify(
        self, question: str, candidate: str, *, context: str = "",
        evidence: str = "", task_type: str = "auto",
    ) -> VerificationResult:
        """Judge one SQL or Python/pandas candidate without executing it.

        Supply SQL DDL/dialect or table schemas/samples/runtime details in context.
        Expected generation/format failures return status='failed' after retries;
        runtime, programming, and cache errors propagate to the caller.
        """
        return self.verify_candidates(
            question, [candidate], context=context, evidence=evidence,
            task_type=task_type,
        )[0]

    def verify_candidates(
        self, question: str, candidates: Sequence[str], *, context: str = "",
        evidence: str = "", task_type: str = "auto",
    ) -> list[VerificationResult]:
        """Judge candidates for one task independently, preserving input order.

        No clustering or validity filtering is performed.
        An empty candidate list returns [].
        Auto detection uses the same solve(tables) heuristic as the dataset CLI.
        """
        task = build_context({
            "question": question, "context": context, "evidence": evidence,
            "candidates": [{"text": c} for c in candidates],
        }, task_type=task_type)
        clusters = tuple(ClusterInput(str(i), c, 1, {}) for i, c in enumerate(candidates))
        q = QuestionInput("", task.question, task.evidence, task.context,
                          "", "", clusters, {}, task.task_type)
        return verify_many(
            [(q, c) for c in clusters], self.backend, self.cache,
            self.retries, self.retry_delay, self.batch_size,
        )
