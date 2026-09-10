"""Prompt templates for naive candidate selection."""
from __future__ import annotations

from typing import Mapping, Sequence


SYSTEM_PROMPT = """You are a careful evaluator that selects the best candidate answer.
Choose exactly one candidate index from the provided list, or return null when none is reliable.
Return JSON only with keys: selected_index, confidence, reason.
- selected_index: integer index into the provided candidate list (0-based), or null to defer to oracle
- confidence: number in [0, 1]
- reason: short explanation
Do not include markdown or extra keys."""


def _candidate_lines(candidates: Sequence[Mapping[str, object]]) -> list[str]:
    lines = []
    for i, candidate in enumerate(candidates):
        text = str(candidate.get("text", "")).strip()
        invalid = bool(candidate.get("invalid_mask", False))
        marker = "invalid=true" if invalid else "invalid=false"
        lines.append(f"[{i}] ({marker})\n{text}")
    return lines


def selector_user_prompt(row: Mapping[str, object]) -> str:
    """Build a row-level prompt with optional context fields."""

    question = str(row.get("question", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    context = str(row.get("context", "")).strip()
    metadata = row.get("metadata")

    parts = [
        "Select the single best candidate answer for this task.",
        f"QUESTION:\n{question}",
    ]
    if evidence:
        parts.append(f"EVIDENCE:\n{evidence}")
    if context:
        parts.append(f"CONTEXT:\n{context}")
    if isinstance(metadata, Mapping) and metadata:
        parts.append(f"METADATA:\n{dict(metadata)}")
    parts.append("CANDIDATES:")
    parts.extend(_candidate_lines(row.get("candidates", [])))
    parts.append("Return JSON only.")
    return "\n\n".join(parts)
