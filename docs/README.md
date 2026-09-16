# Verifier usage

The semantic verifier judges whether a SQL or Python/pandas candidate satisfies
one semantic perspective of a task using the supplied context. It does not execute candidates.

- [Dataset processing](verifier-dataset.md): annotate a saved dataset or JSONL file.
- [Deployment](verifier-deployment.md): reuse the verifier inside an application.
- [Grounded block signals](verifier-block-signals.md): shared requirements,
  local contributions, omission checks, and evidence for downstream EM.

Both paths use the same verification engine and support four separately assessed perspectives:

| Label field | What it assesses |
| --- | --- |
| `output_semantics` | Requested values, columns, aggregations, and output structure |
| `constraint_semantics` | Filters, comparisons, conversions, and transformation rules |
| `ranking_semantics` | Ordering, extrema, top-k, limits, and ties |
| `relation_semantics` | Joins, keys, grouping, reshaping, and row preservation |

Labels are `S` (satisfied), `U` (unresolved), `C` (contradicted), and `NA`
(not applicable). No overall correctness verdict is produced. Check the result
status before interpreting labels: skipped dataset candidates have null labels.
Verification errors propagate immediately, without retry or fallback labels. These judgments are model assessments, not
execution results or ground-truth correctness labels.

Run commands from the repository root. For a fresh, self-contained environment:

```bash
uv sync --locked
```

The guides use `uv run`. If using the existing workspace environment with shared
system packages, use `uv run --no-sync` instead, as described in the
[project README](../README.md).
