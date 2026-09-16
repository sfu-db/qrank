# Cluster judge

This command asks `Qwen/Qwen2.5-Coder-7B-Instruct` to select one valid SQL
agreement cluster per BIRD question. It excludes invalid/empty candidates before
building clusters, includes only the first valid representative SQL from each
cluster, and never puts candidate sources, agreement counts, or labels in the
prompt. Rows with no valid candidates skip inference.

Run a small pilot first:

```bash
uv run --no-sync python -m qrank.judge \
  --config training_configs/judge_qwen2p5_coder_7b_vllm.json \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output dataset/bird_dev_qwen_judged_pilot \
  --limit 10
```

For the complete dataset, omit `--limit` and use a new output directory. Schema
discovery automatically finds the bundled BIRD `schema.json`; `--schemas`,
`--db-root`, and `--descriptions-root` can override or enrich it.

The output is a Hugging Face `DatasetDict` with all original columns plus a
`qwen_judge` column. Use `--column NAME` to change that name. Each annotation
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
