"""Response format for one independent semantic perspective."""
from __future__ import annotations

import json

from .models import VerificationError

PERSPECTIVES = ("output_semantics", "constraint_semantics", "ranking_semantics", "relation_semantics")
VERDICTS = {"S", "U", "C", "NA"}


def response_schema(perspective: str) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["reason", "label"],
        "properties": {
            "reason": {"type": "string", "minLength": 1},
            "label": {"type": "string", "enum": ["S", "U", "C"] if perspective == "output_semantics" else ["S", "U", "C", "NA"]},
        },
    }


def parse_response(raw: str, perspective: str) -> tuple[str, str]:
    result = json.loads(raw)
    label = result["label"]
    if label not in VERDICTS or (perspective == "output_semantics" and label == "NA"):
        raise VerificationError(f"Invalid label for {perspective}: {label!r}")
    return label, result["reason"]
