"""Response format for one independent semantic perspective."""
from __future__ import annotations

import json
import copy

from .models import QuestionInput, RequirementSet, VerificationError

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
    validate_schema(result, response_schema(perspective))
    label = result["label"]
    if label not in VERDICTS or (perspective == "output_semantics" and label == "NA"):
        raise VerificationError(f"Invalid label for {perspective}: {label!r}")
    return label, result["reason"]


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


TEXT = {"type": "string", "minLength": 1}
OPTIONAL_TEXT = {"type": ["string", "null"], "minLength": 1}
# vLLM 0.11's xgrammar backend rejects uniqueItems; enforce uniqueness in parsing.
STRINGS = {"type": "array", "items": TEXT}
SOURCE_SCHEMA = object_schema({
    "field": {"type": "string", "enum": ["question", "evidence", "schema"]},
    "quote": TEXT,
})
SOURCES_SCHEMA = {"type": "array", "items": SOURCE_SCHEMA}
REQUIREMENTS_SCHEMA = object_schema({
    "requirements": {"type": "array", "minItems": 1, "items": object_schema({
        "requirement_id": TEXT,
        "dimension": {"type": "string", "enum": list(PERSPECTIVES)},
        "text": TEXT,
        "source_refs": {**SOURCES_SCHEMA, "minItems": 1},
        "ambiguity": OPTIONAL_TEXT,
    })},
    "unresolved": STRINGS,
})
GROUNDED_SCHEMA = object_schema({
    "label": {"type": "string", "enum": ["S", "U", "C", "NA"]},
    "reason": TEXT,
    "requirement_ids": STRINGS,
    "source_refs": SOURCES_SCHEMA,
    "sql_evidence": {"type": "array", "items": object_schema({
        "scope_id": TEXT, "section": TEXT,
        "operation_index": {"type": "integer", "minimum": 0},
        "fragment": TEXT,
    })},
    "inspected_scope_ids": STRINGS,
    "missing_behavior": OPTIONAL_TEXT,
    "conflict_type": OPTIONAL_TEXT,
    "unresolved_reason": OPTIONAL_TEXT,
    "applicability_reason": OPTIONAL_TEXT,
})


def grounded_response_schema(target: dict, blocks: list[dict]) -> dict:
    """Constrain label-dependent fields and assignments before generation.

    Mutually exclusive label branches are supported by the installed xgrammar
    backend. Reference membership and raw quotes are still checked by parsing.
    """
    coverage = target["observation_kind"] == "requirement_coverage"
    allowed = target["allowed_requirement_ids"]
    labels = ["S", "U", "C"] if coverage else ["S", "U", "C", "NA"]
    if not coverage and (not allowed or not any(target["sections"].values())):
        labels = ["NA"]
    elif target["scope_role"] == "root_result" and target["perspective"] == "output_semantics":
        labels = ["S", "U", "C"]
    branches = []
    for label in labels:
        properties = copy.deepcopy(GROUNDED_SCHEMA["properties"])
        properties["label"]["enum"] = [label]
        for name in ("missing_behavior", "conflict_type", "unresolved_reason", "applicability_reason"):
            properties[name] = {"type": "null"}
        properties["requirement_ids"] = ({"type": "array", "items": {"type": "string", "enum": allowed},
                                         "minItems": 1 if label != "NA" else 0,
                                         "maxItems": len(allowed)} if allowed
                                        else {"type": "array", "items": TEXT, "maxItems": 0})
        properties["inspected_scope_ids"] = {
            "type": "array", "minItems": 1,
            "items": {"type": "string", "enum": [b["scope_id"] for b in blocks if b["reachable"]]},
        }
        if label in {"S", "C"}:
            properties["source_refs"]["minItems"] = 1
            if label == "S" or not coverage:
                properties["sql_evidence"]["minItems"] = 1
            if not coverage:
                # First cite the target contribution; subsequent entries may
                # describe dependencies. The decoder receives the same contract.
                selected = {name: values for name, values in target["sections"].items() if values}
                properties["sql_evidence"]["prefixItems"] = [object_schema({
                    "scope_id": {"type": "string", "enum": [target["scope_id"]]},
                    "section": {"type": "string", "enum": list(selected)},
                    "operation_index": {"type": "integer", "minimum": 0,
                                        "maximum": max(map(len, selected.values())) - 1},
                    "fragment": TEXT,
                })]
        if label == "C":
            properties["conflict_type"] = TEXT
            if coverage:
                properties["missing_behavior"] = OPTIONAL_TEXT
        elif label == "U":
            properties["unresolved_reason"] = TEXT
        elif label == "NA":
            properties["applicability_reason"] = TEXT
        branches.append(object_schema(properties))
    return {"anyOf": branches}


def validate_schema(value, schema: dict, path: str = "response") -> None:
    """Enforce the small JSON-schema subset used here even on unconstrained backends."""
    if "anyOf" in schema:
        # Our unions are explicitly discriminated by label; select the branch
        # directly rather than attempting parses and swallowing failures.
        label = value.get("label") if isinstance(value, dict) else None
        branches = [branch for branch in schema["anyOf"]
                    if label in branch["properties"]["label"]["enum"]]
        if len(branches) != 1:
            raise VerificationError(f"{path}: no schema branch for label {label!r}")
        validate_schema(value, branches[0], path)
        return
    kinds = schema["type"]
    kinds = kinds if isinstance(kinds, list) else [kinds]
    types = {"object": dict, "array": list, "string": str, "integer": int, "null": type(None)}
    if not any(type(value) is types[kind] for kind in kinds):
        raise VerificationError(f"{path}: expected {kinds}")
    if value is None:
        return
    if "enum" in schema and value not in schema["enum"]:
        raise VerificationError(f"{path}: invalid value {value!r}")
    if isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        raise VerificationError(f"{path}: expected nonempty text")
    if type(value) is int and value < schema.get("minimum", value):
        raise VerificationError(f"{path}: below minimum")
    if type(value) is int and value > schema.get("maximum", value):
        raise VerificationError(f"{path}: above maximum")
    if isinstance(value, dict):
        if set(value) != set(schema["properties"]):
            raise VerificationError(f"{path}: expected exactly {list(schema['properties'])}")
        for key, child in schema["properties"].items():
            validate_schema(value[key], child, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise VerificationError(f"{path}: too few items")
        if len(value) > schema.get("maxItems", len(value)):
            raise VerificationError(f"{path}: too many items")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise VerificationError(f"{path}: duplicate items")
        for i, item in enumerate(value):
            prefix = schema.get("prefixItems", [])
            validate_schema(item, prefix[i] if i < len(prefix) else schema["items"], f"{path}[{i}]")


def validate_sources(refs: list[dict], q: QuestionInput) -> None:
    for ref in refs:
        if ref["quote"] not in getattr(q, ref["field"]):
            raise VerificationError(f"Source quote not present in {ref['field']}: {ref['quote']!r}")


def parse_requirements(raw: str, q: QuestionInput) -> dict:
    result = json.loads(raw)
    validate_schema(result, REQUIREMENTS_SCHEMA)
    for i, requirement in enumerate(result["requirements"], 1):
        if requirement["requirement_id"] != f"r{i}":
            raise VerificationError("Requirements must use contiguous IDs r1, r2, ...")
        validate_sources(requirement["source_refs"], q)
    if not any(r["dimension"] == "output_semantics" for r in result["requirements"]):
        raise VerificationError("Requirements must include the requested output, preserving any ambiguity")
    return RequirementSet(tuple(result["requirements"]), tuple(result["unresolved"])).as_dict()


def parse_grounded(raw: str, *, q: QuestionInput, sql: str, requirements: dict,
                   blocks: list[dict], target: dict) -> dict:
    result = json.loads(raw)
    validate_schema(result, GROUNDED_SCHEMA)
    for field in ("requirement_ids", "inspected_scope_ids"):
        if len(result[field]) != len(set(result[field])):
            raise VerificationError(f"{field}: duplicate references")
    known = {r["requirement_id"] for r in requirements["requirements"]}
    if not set(result["requirement_ids"]) <= known:
        raise VerificationError("Observation references an unknown requirement")
    if not set(result["requirement_ids"]) <= set(target["allowed_requirement_ids"]):
        raise VerificationError("Observation references a requirement outside its assigned responsibility")
    label = result["label"]
    if label != "NA" and not result["requirement_ids"]:
        raise VerificationError("Applicable observations require requirement references")
    coverage = target["observation_kind"] == "requirement_coverage"
    if coverage and (result["requirement_ids"] != [target["requirement_id"]] or label == "NA"):
        raise VerificationError("Coverage must check exactly its assigned requirement using S/U/C")
    if (not coverage and target["scope_role"] == "root_result"
            and target["perspective"] == "output_semantics" and label == "NA"):
        raise VerificationError("The root output responsibility always applies")
    validate_sources(result["source_refs"], q)
    scopes = {b["scope_id"]: b for b in blocks if b["reachable"]}
    inspected = set(result["inspected_scope_ids"])
    if not inspected or not inspected <= scopes.keys() or target["scope_id"] not in inspected:
        raise VerificationError("Inspected scopes must be reachable and include the target scope")
    target_evidence = False
    for evidence in result["sql_evidence"]:
        scope_id, section = evidence["scope_id"], evidence["section"]
        if scope_id not in inspected:
            raise VerificationError("SQL evidence must belong to an inspected scope")
        operations = scopes[scope_id]["sections"].get(section, [])
        if evidence["operation_index"] >= len(operations):
            raise VerificationError("SQL evidence references a nonexistent AST operation")
        if evidence["fragment"] not in sql:
            raise VerificationError("SQL evidence fragment must occur exactly in the raw candidate")
        target_evidence |= scope_id == target["scope_id"] and section in target["sections"]
    if label in {"S", "C"} and not result["source_refs"]:
        raise VerificationError("Both S and C require source evidence")
    if label == "S" and not result["sql_evidence"]:
        raise VerificationError("S requires SQL evidence")
    missing = result["missing_behavior"] is not None
    if missing and (not coverage or label != "C" or result["conflict_type"] != "missing_required_behavior"):
        raise VerificationError("Missing behavior belongs to a C coverage check, once per requirement")
    if missing and inspected != scopes.keys():
        raise VerificationError("An omission check must inspect all reachable scopes")
    if label == "C" and (not result["conflict_type"] or not (result["sql_evidence"] or missing)):
        raise VerificationError("C requires a conflict type and SQL evidence or explicit missing behavior")
    if result["conflict_type"] == "missing_required_behavior" and not missing:
        raise VerificationError("Missing-required-behavior conflicts require an omission record")
    if not coverage and label in {"S", "C"} and not target_evidence:
        raise VerificationError("Local S/C must cite an operation in the target block")
    if (label == "U") != (result["unresolved_reason"] is not None):
        raise VerificationError("Only U must supply an unresolved reason")
    if (label == "NA") != (result["applicability_reason"] is not None):
        raise VerificationError("Only NA must supply an applicability reason")
    if label != "C" and result["conflict_type"] is not None:
        raise VerificationError("Only C may supply a conflict type")
    validate_schema(result, grounded_response_schema(target, blocks))
    return result
