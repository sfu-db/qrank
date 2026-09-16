"""Compare saved overall-verifier judgments with aligned benchmark labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from datasets import load_from_disk


VERDICTS = ("S", "U", "C", "NA")


def analyze(path: str, split: str, limit: int | None = None) -> dict:
    rows = load_from_disk(path)[split]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows.select(range(min(limit, len(rows))))
    table = Counter()
    unique_table = Counter()
    row_counts = Counter()
    source_tables: dict[str, Counter] = {}
    unknown_truth = 0

    for row in rows:
        candidates = row["candidates"]
        truths = row["labels"]
        judgments = row["semantic_verifier_overall"]
        if not (len(candidates) == len(truths) == len(judgments)):
            raise ValueError(f"Misaligned row {row['question_id']}")
        seen = set()
        assessed = []
        for candidate, truth, judgment in zip(candidates, truths, judgments, strict=True):
            verdict = judgment["label"]
            if verdict is None:
                continue
            if verdict not in VERDICTS:
                raise ValueError(f"Unknown verdict {verdict!r} on row {row['question_id']}")
            if truth is None:
                unknown_truth += 1
                continue
            if not isinstance(truth, bool):
                raise ValueError(f"Non-boolean benchmark label on row {row['question_id']}")
            key = (truth, verdict)
            table[key] += 1
            source_tables.setdefault(candidate["source"], Counter())[key] += 1
            judgment_key = (judgment["cache_key"], judgment["identifier"])
            if judgment_key not in seen:
                unique_table[key] += 1
                seen.add(judgment_key)
            assessed.append((truth, verdict))

        if not assessed:
            row_counts["no_assessed_candidate"] += 1
            continue
        row_counts["assessed"] += 1
        has_positive = any(truth for truth, _ in assessed)
        s_truths = [truth for truth, verdict in assessed if verdict == "S"]
        row_counts["positive_available"] += has_positive
        row_counts["all_S"] += len(s_truths) == len(assessed)
        row_counts["all_S_no_positive"] += len(s_truths) == len(assessed) and not has_positive
        row_counts["any_S"] += bool(s_truths)
        row_counts["positive_S_available"] += any(s_truths)
        row_counts["S_only_negative"] += bool(s_truths) and not any(s_truths)
        row_counts["positive_available_S_only_negative"] += (
            has_positive and bool(s_truths) and not any(s_truths)
        )
        row_counts["positive_available_no_S"] += has_positive and not s_truths
        row_counts["single_S"] += len(s_truths) == 1
        row_counts["single_S_positive"] += len(s_truths) == 1 and s_truths[0]

    def as_table(counts: Counter) -> dict:
        return {
            "benchmark_positive": {v: counts[(True, v)] for v in VERDICTS},
            "benchmark_negative": {v: counts[(False, v)] for v in VERDICTS},
        }

    return {
        "input": path,
        "split": split,
        "questions": len(rows),
        "assessed_candidates": sum(table.values()),
        "assessed_with_unknown_benchmark_label": unknown_truth,
        "candidate_table": as_table(table),
        "distinct_judgment_table": as_table(unique_table),
        "question_counts": dict(sorted(row_counts.items())),
        "source_tables": {source: as_table(counts) for source, counts in sorted(source_tables.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Saved overall-verifier DatasetDict")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N questions, for a matched pilot comparison")
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.split, args.limit), indent=2))


if __name__ == "__main__":
    main()
