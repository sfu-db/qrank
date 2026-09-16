# Dataset verification

The verifier estimates S/U/C/NA independently for output, constraint, ranking,
and relation semantics. It does not produce an overall correctness verdict.
Each inference request contains one candidate and one perspective's rubric.
SQL requests include both raw SQL and a deterministic SQLGlot AST summary.

```bash
uv run qrank-verifier transform-dataset \
  --config training_configs/verifier_qwen3_8b.json \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output dataset/processed/bird-perspective-pilot \
  --limit 5
```

Use `--perspective constraint_semantics` to evaluate only that perspective.
Omitting it evaluates all four in independent requests. `--batch-size` controls
perspective requests per inference batch, not candidate count. For a full run,
omit `--limit` and choose a new output directory.

## Inputs and context

Input can be a saved Hugging Face DatasetDict or JSONL (loaded into `--split`,
default `test`). Each row needs a unique nonempty `question_id`, a `question`,
and candidates with `text`. Optional `evidence` and `context` supply task details.
Candidates marked `invalid_mask=true` or with empty text are explicitly skipped.

SQL rows require a symmetric, transitive N-by-N `agreement` matrix. The original
agreement grouping is preserved as `agreement_cluster_id`. Distinct stripped SQL
texts within a group are verified separately. Only identical text under the same
task/context shares judgments; an execution agreement does not establish that
SQL-specific reasons apply to every member.

SQL schemas can come from `--schemas` (a database-to-DDL JSON mapping or BIRD
`dev_tables.json`) or `--db-root` (read-only SQLite schema discovery). Otherwise,
discovery searches repository `dataset/base` for `schema.json`, then
`dev_tables.json`, then SQLite files. Multiple matching schema files require an
explicit path. Database IDs come from `metadata.db_id`, then `group_id`.
If no schemas are discovered, supplied row context is used.

`--db-root` also loads documented column/value descriptions from
`<db_id>/database_description/*.csv`, with source paths retained. Use
`--descriptions-root` to supply that directory tree alongside `--schemas` or
row context. These descriptions are supplied to requirement extraction and
candidate checking; sample values are not treated as exhaustive domains.

Use `--sql-dialect sqlite` for explicit schema inputs. Automatic BIRD discovery
and `--db-root` set SQLite. Row/metadata `sql_dialect` is also supported; the CLI
option takes precedence. The application API can use `sql_dialect="sqlite"`.
Without a dialect, SQLGlot uses its generic parser; supply the real dialect for
accurate interpretation.

AST summaries preserve query blocks, CTE sources, projections, joins,
aggregations, Boolean filters, grouping, HAVING, windows, ordering and limits.
They describe operations; they neither define the task's requirements nor prove
SQL execution validity or semantic correctness. Raw SQL remains in the prompt.
Parse errors stop the run; no SQL repair or parser fallback is attempted.

PrepBench uses `--task-type prepbench` or auto-detection from task metadata or a
`def solve(tables...)` signature. It receives raw Python/pandas code and task
context, without a SQL AST. If agreement is absent, candidate `cluster_id` values
supply groups. `--prepbench-root` reads headers and up to five CSV records from
`<root>/<question_id>/inputs/*.csv`. Samples do not establish runtime dtypes.
Missing files under an explicit root raise errors. Gold outputs are never read.

## Outputs and errors

Original columns and candidate order are preserved. Each `semantic_verifier`
entry contains `cluster_id`, `agreement_cluster_id`, `status`, `cache_key`,
`labels`, `reasons`, `block_results`, `requirement_results`, and `requirements`.
For SQL, `requirements` is a JSON string containing the shared source-grounded
contract; block and coverage results retain separate local weak signals.
Invalid candidates have empty result lists and an empty requirements string.
Labels contain only root block signals for the requested perspectives; there
is no `overall`. Status is `ok`, `cached`, or `invalid`. Skipped candidates have
null labels, not semantic NA judgments.

```python
from datasets import load_from_disk

row = load_from_disk("dataset/processed/bird-perspective-pilot")["test"][0]
for candidate, annotation in zip(row["candidates"], row["semantic_verifier"], strict=True):
    print(candidate["text"], annotation["status"], annotation["labels"])
```

vLLM constrains responses using xgrammar, without decoder fallback. SQL extracts
requirements before candidate checks and requires structured evidence for S/C,
explicit missing-behavior records, and separate U/NA reasons. See
[grounded block signals](verifier-block-signals.md) for the record contract.
PrepBench retains `{"label": "S|U|C|NA", "reason": "..."}` with output NA excluded.
Other text backends receive the schema in the prompt. Parsers enforce the schema
and evidence references regardless of backend.
JSON, backend, SQL parsing, and cache errors propagate directly: no retries,
JSON repair, catch-and-rethrow wrappers, or failure-to-NA conversion.

`verifier_trace.jsonl` records inputs before SQL parsing, requests before inference,
and raw completions before JSON parsing. A failed run does not publish a completed
dataset or success manifest. Fix the error and rerun to reuse successful cached
requirements and judgments. The removed `--retries` and `--retry-delay` flags are rejected.

`semantic_verifier.manifest.json` records prompt version, backend identity,
selected perspectives, verification unit and counts. Label counts are per
verified distinct-text group within each agreement cluster. They exclude skipped
candidates. `n_agreement_clusters` separately counts the original groups.
Completed runs have zero failed or unverified groups.

Cache defaults to `<output>/verifier_cache.sqlite`; use `--cache` to reuse another
cache. Check keys include the prompt, raw candidate, AST summary (including
parser version/dialect), fixed contract, context, output schema, and backend identity.
Extraction keys omit candidates and use their own prompt version. v13 does not
reuse previous prompts' judgments. A completed DatasetDict cannot be
overwritten; choose a new `--output` to reprocess it.

Model and batching options come from the required JSON config; CLI flags override
those values. Paths are relative to the working directory. Use
`qrank-verifier transform-dataset --help` for all flags. Reduce batch size for GPU
memory errors; keep prompt plus response within the configured context window.
