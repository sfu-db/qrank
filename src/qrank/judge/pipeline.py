"""Cluster candidates, ask the model once per question, and record its choice."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from qrank.llms import LLMBackend

from qrank.verifier.context import detect_task_type

from .prompts import candidate_identifier, judge_user_prompt, system_prompt


@dataclass(frozen=True)
class JudgeResult:
    question_id: str
    status: str
    selected_identifier: str | None
    selected_candidate_index: int | None
    selected_candidate_indices: tuple[int, ...]
    selected_sql: str | None
    clusters: tuple[dict[str, Any], ...]
    raw_response: str
    is_correct: bool | None


def _valid_indices(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index for index, candidate in enumerate(candidates)
        if not bool(candidate.get("invalid_mask", False))
        and isinstance(candidate.get("text"), str)
        and bool(candidate["text"].strip())
    ]


def build_clusters(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Group valid candidates by the supplied execution-agreement relation.

    Invalid and empty candidates are removed before grouping and can never become
    representatives. The first valid candidate in dataset order represents a group.
    """
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or any(not isinstance(c, Mapping) for c in candidates):
        raise ValueError("candidates must be a list of objects")
    agreement = row.get("agreement")
    n = len(candidates)
    if not isinstance(agreement, list) or len(agreement) != n or any(
        not isinstance(line, list) or len(line) != n for line in agreement
    ):
        raise ValueError("agreement must be a square matrix aligned with candidates")

    valid = _valid_indices(candidates)
    remaining = set(valid)
    groups: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        members = {seed}
        pending = [seed]
        while pending:
            left = pending.pop()
            for right in valid:
                if bool(agreement[left][right]) != bool(agreement[right][left]):
                    raise ValueError("agreement must be symmetric for valid candidates")
                if bool(agreement[left][right]) and right not in members:
                    members.add(right)
                    pending.append(right)
        if any(not bool(agreement[left][right])
               for left in members for right in members if left != right):
            raise ValueError("agreement must be transitive for valid candidates")
        remaining -= members
        groups.append(sorted(members))

    clusters = []
    for cluster_index, members in enumerate(groups):
        representative = members[0]
        clusters.append({
            "identifier": candidate_identifier(cluster_index),
            "representative_index": representative,
            "candidate_indices": members,
            "representative_sql": candidates[representative]["text"].strip(),
        })
    return tuple(clusters)


def parse_choice(raw: str, identifiers: Sequence[str]) -> str | None:
    """Parse a choice, tolerating Qwen's harmless ``Candidate X`` wrapper."""
    if not isinstance(raw, str):
        raise ValueError("model response must be a string")
    choice = raw.strip().upper()
    if choice == "NONE":
        return None
    wrapped = re.fullmatch(r"CANDIDATE\s+([A-Z]+)", choice)
    if wrapped:
        choice = wrapped.group(1)
    if choice not in identifiers:
        allowed = ", ".join((*identifiers, "NONE"))
        raise ValueError(
            f"model response must be one of {allowed}, optionally formatted as "
            f"'Candidate <identifier>'; got {raw!r}"
        )
    return choice


def _correctness(
    row: Mapping[str, Any], selected_index: int | None, valid_indices: Sequence[int],
) -> bool | None:
    labels = row.get("labels")
    candidates = row.get("candidates", [])
    if not isinstance(labels, list) or len(labels) != len(candidates) or any(
        label is not None and not isinstance(label, bool) for label in labels
    ):
        return None
    known_valid_labels = [labels[index] for index in valid_indices]
    if any(label is None for label in known_valid_labels):
        return None
    if selected_index is None:
        return not any(known_valid_labels)
    return labels[selected_index]


def judge_rows(
    rows: Sequence[Mapping[str, Any]], contexts: Sequence[str], backend: LLMBackend,
    *, batch_size: int = 8, task_types: Sequence[str] | None = None,
) -> list[JudgeResult]:
    """Judge rows in batches while preserving rows with no valid candidates."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(contexts) != len(rows):
        raise ValueError("contexts must align with rows")
    if task_types is None:
        task_types = [detect_task_type(row) for row in rows]
    if len(task_types) != len(rows):
        raise ValueError("task_types must align with rows")

    row_clusters: list[tuple[dict[str, Any], ...]] = []
    requests: list[dict[str, Any]] = []
    request_rows: list[int] = []
    for row_index, (row, context, task_type) in enumerate(
        zip(rows, contexts, task_types, strict=True)
    ):
        clusters = build_clusters(row)
        row_clusters.append(clusters)
        if not clusters:
            continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        qid = str(row.get("question_id", row_index))
        identifiers = [cluster["identifier"] for cluster in clusters]
        requests.append({
            "question_id": qid,
            "cluster_id": "judge",
            "system": system_prompt(task_type),
            "user": judge_user_prompt(
                question=question,
                context=context,
                evidence=str(row.get("evidence") or ""),
                clusters=clusters,
                task_type=task_type,
            ),
            # vLLM uses this as a constrained-decoding choice set. The model
            # cannot emit prose such as "Candidate B" when this is honored.
            "response_choices": [*identifiers, "NONE"],
        })
        request_rows.append(row_index)

    responses: dict[int, str] = {}
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        generated = backend.generate_batch(batch)
        if len(generated) != len(batch):
            raise ValueError(f"backend returned {len(generated)} responses for {len(batch)} requests")
        responses.update(zip(request_rows[start:start + batch_size], generated, strict=True))

    results = []
    for row_index, (row, clusters) in enumerate(zip(rows, row_clusters, strict=True)):
        qid = str(row.get("question_id", row_index))
        valid_indices = [index for cluster in clusters for index in cluster["candidate_indices"]]
        if not clusters:
            results.append(JudgeResult(
                qid, "no_valid_candidates", None, None, (), None, clusters, "",
                _correctness(row, None, valid_indices),
            ))
            continue
        raw = responses[row_index]
        selected = parse_choice(raw, [cluster["identifier"] for cluster in clusters])
        cluster = next((item for item in clusters if item["identifier"] == selected), None)
        selected_index = None if cluster is None else cluster["representative_index"]
        results.append(JudgeResult(
            qid,
            "none" if cluster is None else "selected",
            selected,
            selected_index,
            () if cluster is None else tuple(cluster["candidate_indices"]),
            None if cluster is None else cluster["representative_sql"],
            clusters,
            raw,
            _correctness(row, selected_index, valid_indices),
        ))
    return results


def select_models(row: Mapping[str, Any], models: Sequence[str]) -> dict[str, Any]:
    """Filter a row and slice its known agreement matrix to valid input models.

    No candidate SQL is executed. Model names locate candidates in the input;
    the existing ``invalid_mask`` and empty-text signal determine which located
    candidates are omitted from the rebuilt row.
    """
    if not models or any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError("models must contain at least one non-empty model name")
    if len(set(models)) != len(models):
        raise ValueError("models must not contain duplicates")
    candidates_value = row.get("candidates")
    if not isinstance(candidates_value, list) or any(
        not isinstance(candidate, Mapping) for candidate in candidates_value
    ):
        raise ValueError("candidates must be a list of objects")
    candidates = list(candidates_value)
    n = len(candidates)
    input_agreement = row.get("agreement")
    if not isinstance(input_agreement, list) or len(input_agreement) != n or any(
        not isinstance(line, list) or len(line) != n for line in input_agreement
    ):
        raise ValueError("agreement must be a square matrix aligned with candidates")
    by_source: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        source = candidate.get("source")
        if isinstance(source, str):
            by_source.setdefault(source, []).append(index)
    located: list[int] = []
    for model in models:
        matches = by_source.get(model, [])
        if len(matches) != 1:
            raise ValueError(
                f"question {row.get('question_id')!r}: expected exactly one candidate "
                f"from model {model!r}; found {len(matches)}"
            )
        located.append(matches[0])
    selected = [
        index for index in located
        if not bool(candidates[index].get("invalid_mask", False))
        and isinstance(candidates[index].get("text"), str)
        and bool(candidates[index]["text"].strip())
    ]

    rebuilt = dict(row)
    rebuilt["candidates"] = [dict(candidates[index]) for index in selected]
    rebuilt["agreement"] = [
        [int(bool(input_agreement[left][right])) for right in selected]
        for left in selected
    ]
    for field in ("labels", "semantic_verifier"):
        values = row.get(field)
        if isinstance(values, list) and len(values) == n:
            rebuilt[field] = [values[index] for index in selected]
    return rebuilt
