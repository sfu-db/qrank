"""Historical whole-requirement prompt example; the active workflow uses prompts.py."""
from __future__ import annotations

import json

from .models import RequirementSet


REQUIREMENTS_VERSION = "qrank-svem-requirements-v1"
CHECK_VERSION = "qrank-svem-candidate-check-v1"

REQUIREMENTS_SYSTEM = r"""
You extract atomic semantic obligations for a text-to-SQL task.
Read only the question, evidence/hints, and schema/context. Do not see or imagine any candidate SQL.
Do not invent requirements from common sense. If schema meaning is ambiguous, phrase only what is
actually supported. Split requirements so each item can be checked independently.

Use exactly these dimensions:
- output_semantics: requested returned entity/value/attribute/aggregate/result shape
- constraint_semantics: required predicates/population/literals/ranges/dates/thresholds/denominators
- ranking_semantics: ordering/extreme/top-k target, direction, limit, material NULL behavior
- relation_semantics: required entity/table relationship, join/grouping relationship when established

If a dimension has no task requirement, emit no requirement for that dimension. That absence is the
question-level NA mask later. Output semantics will normally have at least one requirement.
Return JSON only, with short requirement texts and contiguous IDs r1, r2, ... .
""".strip()


def requirements_user(*, question: str, evidence: str, schema: str) -> str:
    example = {
        "requirements": [
            {"requirement_id": "r1", "dimension": "output_semantics", "text": "return the requested count"},
            {"requirement_id": "r2", "dimension": "constraint_semantics", "text": "restrict records to year 2020"},
        ]
    }
    return f"""Question:\n{question}\n\nEvidence/hints:\n{evidence or '(none)'}\n\nSchema/context:\n{schema or '(none)'}\n\nReturn exactly this JSON shape (example values are illustrative):\n{json.dumps(example, ensure_ascii=False)}"""


CHECK_SYSTEM = r"""
You verify ONE candidate SQL against a fixed list of task requirements extracted before seeing this SQL.
Do not reinterpret, add, remove, or rewrite the requirements. Judge SQL behavior, not preferred SQL style.

For each requirement return exactly one label:
S = the SQL behavior satisfies the requirement.
U = the requirement matters, but the supplied SQL/schema evidence is insufficient to establish S or C.
C = a concrete, result-changing conflict is demonstrable.

C has a high burden of proof. For C you MUST provide:
- sql_fragment: an exact short fragment from the candidate SQL that grounds the conflict;
- conflict_type: a compact factual type such as wrong_attribute, wrong_literal, wrong_operator,
  missing_required_behavior, wrong_ranking_direction, wrong_limit, incompatible_relation, wrong_output.
If you cannot point to a concrete SQL fragment/omission and a fixed requirement, use U, not C.

For U provide unresolved_reason. For S, sql_fragment may be provided as supporting evidence.
Do not return NA here: NA is determined deterministically by whether a dimension had zero requirements.
Return JSON only. Do not return an overall label; code will aggregate it deterministically.
""".strip()


def check_user(*, requirements: RequirementSet, sql: str, schema: str) -> str:
    reqs = requirements.as_dict()["requirements"]
    checks = [
        {
            "requirement_id": r["requirement_id"],
            "label": "S|U|C",
            "sql_fragment": None,
            "conflict_type": None,
            "unresolved_reason": None,
        }
        for r in reqs
    ]
    return f"""Fixed requirements:\n{json.dumps(reqs, ensure_ascii=False)}\n\nSchema/context:\n{schema or '(none)'}\n\nCandidate SQL:\n{sql}\n\nReturn exactly:\n{json.dumps({'checks': checks}, ensure_ascii=False)}"""


def retry_user(original_user: str, error: str) -> str:
    return f"""{original_user}\n\nYour previous response was rejected by the strict parser:\n{error}\nReturn ONLY a corrected JSON object matching the requested schema. Do not add prose or markdown fences."""
