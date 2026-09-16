# Naive LLM Selector Baseline

This baseline picks one candidate per question using a local LLM (default: `Qwen/Qwen3-1.7B` via Transformers).

It supports an **oracle deferral** mode:
- The LLM may return `selected_index: null` when uncertain.
- Low-confidence predictions can also be routed to oracle.
- Each configurable oracle budget is allocated to deferrals first, then the
  lowest-confidence selections.

The output includes per-row selections and a manifest with accuracy metrics (when `labels` are available).

## Run

From repo root:

```bash
uv run --no-sync qrank-selector --config training_configs/selector_naive_qwen3_1p7b.json
```

Or via module:

```bash
uv run --no-sync python -m qrank.baselines.selector --config training_configs/selector_naive_qwen3_1p7b.json
```

For PrepBench config:

```bash
uv run --no-sync qrank-selector --config training_configs/selector_naive_qwen3_1p7b_prepbench.json
```

## Config Schema

The required `--config` argument selects a single-dataset JSON object in
`training_configs`. Include all seven LLM fields shown below; model settings
have no implicit Python defaults.

```json
{
  "output_dir": "dataset/processed/selector_naive_qwen3_1p7b",
  "backend": "transformers",
  "model": "Qwen/Qwen3-1.7B",
  "device": "auto",
  "dtype": "auto",
  "temperature": 0.0,
  "max_tokens": 128,
  "revision": "main",
  "batch_size": 2,
  "oracle_budget_ratio": 0.2,
  "oracle_budget_ratios": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
  "oracle_budget_count": null,
  "show_progress": true,
  "progress_every": 20,
  "overwrite": false,
  "dataset": {
    "name": "bird_dev_agr_q_sql_af_AQXRO_ss_m5",
    "path": "dataset/processed/bird_dev_agr_q_sql_af_AQXRO_ss_m5",
    "split": "test",
    "limit": 50
  }
}
```

Field notes:
- `revision`: Hugging Face revision for model/tokenizer (`main`, tag, branch, or commit hash).
- `oracle_budget_ratio`: fraction of rows allocated to oracle, rounded down to
  a whole number of calls and subject to `oracle_budget_count`.
- `oracle_budget_ratios`: optional list of fractions for a budget sweep. Each
  budget reuses the same model responses; inference runs only once. The single
  `oracle_budget_ratio` still controls the primary `selections.jsonl` result.
- `oracle_budget_count`: optional hard cap on oracle calls; effective budget is
  `min(floor(n_rows * oracle_budget_ratio), oracle_budget_count)` when count is set.
- `dataset.limit`: optional pilot size.
- `show_progress`: print batch progress to stderr during generation.
- `progress_every`: update progress every N batches (default `20`).
- `overwrite`: existing nonempty output directories are rejected before model loading.
  Set to `true` to replace a previous run, or choose a new `output_dir`.

## Output Files

For each run, the baseline writes into:

`<output_dir>/<dataset.name>/`

Files:
- `selections.jsonl`: one row per question
- `selector.manifest.json`: run metadata and aggregate metrics
- `selections.budget_<index>.jsonl`: per-row results for each configured sweep
  entry; the manifest's `budget_sweep` maps files to ratios, effective call
  limits, and metrics.

`selections.jsonl` fields:
- `question_id`
- `status`: `ok`, `oracle`, `fallback`, or `no_valid_candidates`
- `selected_index`: original candidate index, or `null` for oracle/unanswered cases
- `selected_text`
- `confidence`
- `reason`
- `valid_indices`
- `requested_oracle`: original selection was null (a deferral or no valid candidate)
- `used_oracle`: this row was allocated an oracle call; can be true even when
  `requested_oracle` is false, because confidence ranking fills the budget
- `final_source`: `llm`, `oracle`, `fallback`, or `none`
- `is_correct`: boolean when labels are available, else `null`
- `has_labels`: whether any ground-truth candidate labels are available for the row
- `raw_response`: model response, including malformed responses for debugging;
  empty when inference was skipped because no candidates were valid

The CLI prints the end-to-end accuracy and manifest path on completion, and
also emits the full manifest as JSON on stdout.

## Compare Oracle Budgets

The BIRD config includes budgets of 0%, 5%, 10%, 20%, 30%, 50%, and 100%.
Run it with the command above to print a budget-versus-accuracy table and save
all scores under `budget_sweep` in the manifest. Adjust `oracle_budget_ratios`
to compare other budgets, or omit it to run only the primary budget.

Each policy starts with the original model selections and a fresh budget.
Null selections (including rows without valid candidates) have first priority,
in dataset order. Remaining calls go to selected answers in ascending order of
self-reported confidence, with dataset order breaking ties. Parse/backend
failures with a valid fallback candidate have confidence zero. Ground-truth
labels are used only for scoring, never for allocation.

Every effective budget is fully used, even if all answers have high confidence.
The same ranking is used for all budgets, so larger budgets include all rows
chosen at smaller budgets. `oracle_budget_count`, if set, caps **every** budget
in the sweep. A 100% budget routes all rows to oracle unless the count cap
restricts it. All oracle scores assume a perfect oracle.

`oracle_confidence_threshold` has been removed; remove it from older configs.
Confidence is still generated for ranking, but is not a calibrated probability
of correctness.

## Accuracy Metrics

Supply a `labels` list aligned to candidates. Entries must be booleans or
`null` for unknown correctness; strings such as `"False"` are rejected before
inference. Semantic-verifier judgments are not ground-truth labels.

Metrics:
- `llm_only_accuracy`: accuracy over rows finalized by LLM/fallback.
  Only rows with known correctness are included; this is conditional accuracy
  after deferral, not a separate run with oracle disabled.
- `end_to_end_accuracy`: accuracy over **all input rows**, including oracle
  successes and unanswered questions (counted as incorrect). It is `null` if
  any final outcome has unknown correctness, or if the split is empty.
  Oracle rows are correct by definition (simulated perfect oracle).
- `end_to_end_denominator`: total input rows.
- `evaluated_accuracy`: accuracy over only outcomes with known correctness.
  Use `evaluation_coverage` and `n_rows_missing_correctness` to see coverage;
  a partial score is not the full end-to-end accuracy.
- `n_rows_with_labels`: rows with at least one ground-truth candidate label;
  simulated oracle success does not imply that labels exist.
- `evaluation_status`: `complete`, `partial`, `unavailable`, or `empty`.
- `oracle_usage_rate`: oracle calls divided by total rows.
- `oracle_requested`: number of original null selections, independent of budget.

The bundled BIRD dataset has ground-truth labels. The bundled PrepBench export
does not; its autonomous selections cannot be scored without evaluation labels.

## Notes

- Invalid/empty candidates are hidden from the model. Displayed indices retain
  their original positions in the candidate list, so they may have gaps.
- Rows without valid candidates skip model inference and request oracle help,
  sharing first priority with other deferrals in dataset order.
- If parsing/model output fails, baseline falls back to first valid candidate.
- If oracle is requested but budget is exhausted, an existing valid selection is
  retained; a null selection falls back to the first valid candidate, if any.
  With no valid candidates, it chooses uniformly at random from all original
  candidates and records `status=fallback`. The random generator is seeded with
  `selector-fallback-v1:<row position>:<question_id>` so fallback choices remain
  reproducible across runs and budget sweeps with the same row order and IDs.
  Correctness comes from the chosen candidate's label. An empty candidate list
  remains unanswered (`status=no_valid_candidates`) unless allocated an oracle.
