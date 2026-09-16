# Application usage

The verifier judges each semantic perspective independently. It does not estimate
whole-candidate correctness or aggregate the perspectives into an overall label.

```python
from qrank.verifier import Verifier, SQLiteCache

verifier = Verifier(
    config_path="training_configs/verifier_qwen3_8b.json",
    cache=SQLiteCache("verifier.sqlite"),  # Optional.
)
result = verifier.verify(
    question="Return the sum of salaries for employees in department Sales.",
    candidate="SELECT SUM(salary) FROM employees WHERE department = 'HR'",
    context="CREATE TABLE employees (salary REAL, department TEXT);",
    sql_dialect="sqlite",
    perspective="constraint_semantics",
)
print(result.labels["constraint_semantics"])
print(result.reasons["constraint_semantics"])
```

Omit `perspective` to request all four. SQL emits a local check per reachable
scope/perspective plus a coverage check for each requirement in those perspectives.
`verify_candidates(question, candidates, ...)` accepts the same options and
returns results in candidate order. Identical candidates share judgments.
Empty candidate lists return `[]`; an empty candidate string raises.

Supported perspectives are `output_semantics`, `constraint_semantics`,
`ranking_semantics`, and `relation_semantics`. A correct sum with an incorrect
population filter can be output S and constraint C. A projection error should
not automatically change the relationship judgment. NA means that a local check
does not apply; U identifies a material missing fact. Root output and assigned
requirement coverage checks always apply.

SQL inputs include raw SQL and a deterministic AST summary with separate blocks
for nested queries/CTEs. Supply the actual SQL dialect explicitly. PrepBench
inputs use raw Python/pandas with schemas, samples and runtime details in
`context`; they do not use SQL parsing. `task_type` supports `auto`, `text-to-sql`,
and `prepbench`. The application API does not discover files or execute code.

## Configuration and results

Construct the verifier once to reuse the loaded backend. Supply either
`config_path` or an already-loaded `LLMBackend`. Imports do not load model weights.
`batch_size` in config defaults to 8; `Verifier(..., batch_size=16)` overrides it.
The size is the number of requests per inference batch. SQL extraction runs once
per question/context before the local and coverage requests. The supplied cache
also reuses contracts across calls. PrepBench retains four code-level requests.

`VerificationResult` exposes `labels`, `reasons`, `raw_text`, `status`,
`cache_key`, `attempts`, `requirements`, `block_results`, and `requirement_results`.
Labels/reasons expose only the root block's requested perspectives; they are
not coverage summaries. `raw_text` maps block/coverage IDs to raw completions.
The requirements record preserves the source-grounded contract, original inputs,
extractor identity, and extraction cache key. Status is `ok` or `cached`; attempts
is one when any fresh extraction or check was needed, zero when all were cached.
The result cache key identifies the bundle; each request is cached separately.
A later all-perspective call reuses an earlier single-perspective judgment.

See [grounded block signals](verifier-block-signals.md) for structured evidence,
omission records, and metadata for downstream EM. Mixed S/C observations are
retained without reconciliation or an overall correctness judgment.

There are no retries, automatic response repairs, fallback labels, or overall
verdicts. SQL parsing, JSON, inference and cache errors propagate to the caller.
The vLLM backend uses xgrammar to constrain structured responses. Transformers
and custom text backends receive the same schema in the prompt without decoding
constraints. Parsing failures remain errors for those backends as well.

Pass `audit_path="var/verifier_trace.jsonl"` to retain prompts and completions;
create the parent directory first. The trace is written before response parsing,
so invalid JSON is available for debugging. Cache is opt-in; keys include the
perspective prompt, SQL/AST, context, response schema and backend identity.

Calls are synchronous. Applications manage workers/concurrency; each built-in
backend serializes generation on its instance. Separate processes load separate
model copies. Keep prompt plus response within model context limits and reduce
batch size if GPU memory is exhausted.
