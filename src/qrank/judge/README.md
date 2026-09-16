# Cluster judge

This command asks the configured judge model to select one SQL agreement cluster
per BIRD question. `--models` specifies which candidate-source models may be
used. For every row, the command looks up those model names, removes candidates
already marked invalid (or containing empty SQL), and rebuilds the agreement
matrix by slicing the input matrix to the remaining candidate indices.

Candidate SQL is never executed. The existing `invalid_mask` and `agreement`
fields are authoritative.

The prompt includes only the first valid representative SQL from each rebuilt
cluster and never includes candidate sources, agreement counts, or labels. Rows
with no valid candidates skip inference.

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

The output is a Hugging Face `DatasetDict` whose selected split contains only
valid candidates from the requested models, the corresponding agreement
submatrix, aligned labels/verifier data, and a new `qwen_judge` column. Use
`--column NAME` to change that name. Each annotation
contains the displayed clusters, selected identifier, representative candidate
index and SQL, all candidate indices in the selected agreement cluster, raw
model response, status, and label-based correctness when labels are available.
The preferred model response is exactly `A`, `B`, ..., or `NONE`. The parser also
accepts Qwen's common `Candidate A` wrapper, but rejects explanations, Markdown,
punctuation, and identifiers that were not displayed.

```python
from datasets import load_from_disk

row = load_from_disk("dataset/bird_dev_qwen_judged_pilot")["test"][0]
print(row["qwen_judge"])
```
