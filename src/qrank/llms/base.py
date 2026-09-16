"""Shared interface for language model backends."""
from __future__ import annotations

from typing import Any, Sequence


class BackendError(RuntimeError):
    """An expected backend configuration or generation failure."""


class LLMBackend:
    backend_name = "abstract"

    def __init__(self, model: str):
        self.model = model

    def generate(self, *, question_id: str, cluster_id: str, system: str, user: str) -> str:
        raise NotImplementedError

    def generate_batch(self, requests: Sequence[dict[str, Any]]) -> list[str]:
        # Response constraints are decoding metadata. Backends with constrained
        # decoding consume them; other backends rely on the supplied prompt.
        metadata = {"response_schema", "response_choices"}
        return [self.generate(**{k: v for k, v in request.items() if k not in metadata})
                for request in requests]

    def identity(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "model": self.model}
