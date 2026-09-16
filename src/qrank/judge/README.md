# Cluster judge

This command asks the configured judge model to select one agreement cluster
per BIRD or PrepBench task. `--models` specifies which candidate-source models
may be used. For every row, the command looks up those model names, removes
candidates already marked invalid (or containing empty text), and rebuilds the
agreement matrix by slicing the input matrix to the remaining candidate indices.

Candidate SQL/Python is never executed. The existing `invalid_mask` and
`agreement` fields are authoritative.

The prompt includes only the first valid representative SQL query or Python
program from each rebuilt cluster and never includes candidate sources,
agreement counts, or labels. Rows with no valid candidates skip inference.

Run a small pilot first:

```bash
uv run --no-sync python -m qrank.judge \
  --config training_configs/judge_qwen2p5_coder_7b_vllm.json \
  --models QWen2.5-7B XiYanSQL-7B rnj-1-8B \
  --input dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7 \
  --output dataset/bird_dev_qwen_judged_pilot \
  --limit 10
```

For the complete dataset, omit `--limit` and use a new output directory. The
judge model still comes from `--config`; `--models` (or its singular alias
`--model`) selects candidate sources, and accepts one or more names. Prompt
context uses `dataset/base/dev_20240627/schema.json` by default. `--schemas`,
`--db-root`, and `--descriptions-root` only customize prompt context and never
execute candidate SQL.

For the bare PrepBench matrix, supply the JSONL export containing task text and
candidate code. The join happens in memory in matrix-model order:

```bash
uv run --no-sync python -m qrank.judge \
  --config training_configs/judge_qwen2p5_coder_7b_vllm.json \
  --models kimi-k2-thinking glm-4.7 qwen3-235b-a22b-thinking-2507 deepseek-v3.2 devstral-2512 \
  --input dataset/raw/matrix_kimi-k2_glm-4.7_qwen3-235b-a22b_deepseek-v3.2_devstral-2512_prepbench_m5 \
  --split test \
  --prepbench-jsonl dataset/raw/prepbench.jsonl \
  --output dataset/processed/prepbench_judged
```

PrepBench is detected from its `setting`/`task_type` fields (or can be forced
with `--task-type prepbench`). It does not load database schemas, SQLite files,
or database descriptions. Without `--prepbench-root`, the prompt explicitly
states that input table schemas and samples are unavailable. If available,
`--prepbench-root PATH` reads only CSV headers and up to five input rows from
`PATH/<case_id>/inputs/*.csv`; it never reads reference outputs. A previously
transformed PrepBench dataset containing `question`, `candidates`, and
`task_type` can be passed directly without `--prepbench-jsonl`.

The output is a Hugging Face `DatasetDict` whose selected split contains only
valid candidates from the requested models, the corresponding agreement
submatrix, aligned labels/verifier data, and a new `qwen_judge` column. Use
`--column NAME` to change that name. Each annotation
contains the displayed clusters, selected identifier, representative candidate
index and query/program text (in the legacy `selected_sql` field), all candidate
indices in the selected agreement cluster, raw model response, status, and
label-based correctness when labels are available.
The preferred model response is exactly `A`, `B`, ..., or `NONE`. The parser also
accepts Qwen's common `Candidate A` wrapper, but rejects explanations, Markdown,
punctuation, and identifiers that were not displayed.

```python
from datasets import load_from_disk

row = load_from_disk("dataset/bird_dev_qwen_judged_pilot")["test"][0]
print(row["qwen_judge"])
```
