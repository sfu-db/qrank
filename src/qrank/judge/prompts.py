"""Prompts for choosing one representative SQL query."""
from __future__ import annotations

from typing import Mapping, Sequence


SYSTEM_PROMPT = """Given the question, database context/schema, evidence, and candidate SQL queries below, determine which single candidate most correctly answers the question. Evaluate the actual semantics of each SQL, including requested output, filters, joins/relations, aggregation, grouping, ordering, and limits. Do not prefer a candidate because of formatting or apparent source. If none correctly answers the question, choose NONE. Return only the candidate identifier or NONE."""


def candidate_identifier(index: int) -> str:
    """Return spreadsheet-style identifiers: A..Z, AA..AZ, and so on."""
    if type(index) is not int or index < 0:
        raise ValueError("candidate index must be a non-negative integer")
    value = index + 1
    parts: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        parts.append(chr(ord("A") + remainder))
    return "".join(reversed(parts))


def judge_user_prompt(
    *, question: str, context: str, evidence: str,
    clusters: Sequence[Mapping[str, object]],
) -> str:
    """Build a source-blind prompt containing one SQL representative per cluster."""
    context_parts = []
    if context.strip():
        context_parts.append("Database context/schema:\n" + context.strip())
    if evidence.strip():
        context_parts.append("Evidence:\n" + evidence.strip())

    parts = [
        "Question:\n" + question.strip(),
        "Evidence / schema context:\n" + "\n\n".join(context_parts),
    ]
    for cluster in clusters:
        parts.append(
            f"Candidate {cluster['identifier']}:\n"
            f"{str(cluster['representative_sql']).strip()}"
        )
    return "\n\n".join(parts)
