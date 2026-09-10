# Naive LLM Selector Baseline

This baseline picks one candidate per question using a local LLM (default: `Qwen/Qwen3-1.7B` via Transformers).

It supports an **oracle deferral** mode:
- The LLM may return `selected_index: null` when uncertain.
- Low-confidence predictions can also be routed to oracle.
- Oracle usage is capped by a configurable budget.

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

The config is a single-dataset JSON object in `training_configs`.

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
  "oracle_confidence_threshold": 0.6,
  "oracle_budget_ratio": 0.2,
  "oracle_budget_count": null,
  "show_progress": true,
  "progress_every": 1,
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
- `oracle_confidence_threshold`: if model confidence is below this value, request oracle.
- `oracle_budget_ratio`: max fraction of rows allowed to use oracle.
- `oracle_budget_count`: optional hard cap on oracle calls; effective budget is
  `min(floor(n_rows * oracle_budget_ratio), oracle_budget_count)` when count is set.
- `dataset.limit`: optional pilot size.
- `show_progress`: print batch progress to stderr during generation.
- `progress_every`: update progress every N batches.

## Output Files

For each run, the baseline writes into:

`<output_dir>/<dataset.name>/`

Files:
- `selections.jsonl`: one row per question
- `selector.manifest.json`: run metadata and aggregate metrics

`selections.jsonl` fields:
- `question_id`
- `status`: `ok`, `oracle`, `fallback`, or `no_valid_candidates`
- `selected_index`: selected candidate index, or `null` for oracle/no-valid cases
- `selected_text`
- `confidence`
- `reason`
- `valid_indices`
- `requested_oracle`
- `used_oracle`
- `final_source`: `llm`, `oracle`, `fallback`, or `none`
- `is_correct`: boolean when labels are available, else `null`

## Accuracy Metrics

When dataset rows contain a boolean `labels` list aligned to candidates:
- `llm_only_accuracy`: accuracy over rows finalized by LLM/fallback.
- `end_to_end_accuracy`: accuracy over finalized rows including oracle.
  Oracle rows are counted as correct by definition (simulated perfect oracle).
- `oracle_usage_rate`: oracle calls divided by total rows.

If `labels` are missing (for example, some PrepBench exports), accuracy fields may be `null` or have zero denominators.

## Notes

- Invalid/empty candidates are excluded from valid selections.
- If parsing/model output fails, baseline falls back to first valid candidate.
- If oracle is requested but budget is exhausted and valid candidates exist, baseline falls back to first valid candidate.
