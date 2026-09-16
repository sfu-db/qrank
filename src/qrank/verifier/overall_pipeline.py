"""Dataset adapter for one overall S/U/C assessment per candidate."""
from __future__ import annotations

from typing import Any, Sequence

from .models import QuestionInput
from .overall import OverallResult


def attach_overall_results(dataset: Any, split: Any, questions: Sequence[QuestionInput],
                           results: Sequence[OverallResult], output: str, split_name: str) -> dict[str, int]:
    from datasets import Features, List, Value

    annotations = []
    counts = {label: 0 for label in ("S", "U", "C", "NA")}
    result_iter = iter(results)
    for source, question in zip(split, questions, strict=True):
        aligned = [{"status": "invalid", "identifier": "", "label": None,
                    "reason": "invalid or empty candidate", "cache_key": "",
                    "agreement_cluster_id": ""}
                   for _ in source["candidates"]]
        for cluster in question.clusters:
            result = next(result_iter)
            counts[result.label] += 1
            annotation = {
                "status": result.status, "identifier": result.identifier,
                "label": result.label, "reason": result.reason,
                "cache_key": result.cache_key,
                "agreement_cluster_id": cluster.metadata["agreement_cluster_id"],
            }
            for index in cluster.metadata["candidate_indices"]:
                aligned[index] = annotation
        annotations.append(aligned)
    if next(result_iter, None) is not None:
        raise ValueError("more overall results than candidate groups")
    column = "semantic_verifier_overall"
    if column in split.column_names:
        split = split.remove_columns(column)
    annotation_type = List({
        "status": Value("string"), "identifier": Value("string"), "label": Value("string"),
        "reason": Value("string"), "cache_key": Value("string"),
        "agreement_cluster_id": Value("string"),
    })
    features = Features({**split.features, column: annotation_type})
    dataset[split_name] = split.add_column(column, annotations).cast(features)
    dataset.save_to_disk(output)
    return counts
