# Transform PrepBench

Merge the saved matrix dataset with the question/code JSONL export and save a
Hugging Face `DatasetDict`:

```bash
uv run --no-sync python scripts/prepbench/transform.py
```

Defaults:

- `--input`: `dataset/raw/matrix_kimi-k2_glm-4.7_qwen3-235b-a22b_deepseek-v3.2_devstral-2512_prepbench_m5`
- `--prepbench-jsonl`: `dataset/raw/prepbench.jsonl`
- `--output`: `dataset/processed-prepbench`
- All input splits and rows are converted.

For a small subset:

```bash
uv run --no-sync python scripts/prepbench/transform.py \
  --split test --limit 5 --output dataset/processed-prepbench-pilot
```

Default paths are relative to the repository. Explicit relative paths resolve
from your working directory. The output directory must not already exist or
be inside the source dataset.

The conversion joins matrix `case_id` to JSONL `question_id`, then matches
candidates by `source` in the matrix's `models` order. It preserves matrix
columns, including agreement, features, IDs, and validity information, and
adds or replaces:

- `question`: task text from JSONL.
- `task_type`: `prepbench`.
- `candidates`: a list of records containing `source`, `text`, `invalid_mask`,
  and float `features` from the aligned matrix `answer_features`.
- `labels`: the matrix labels converted to booleans, preserving nulls.

Missing candidates are allowed only for models marked invalid in the matrix.
Empty code and JSONL invalid flags also mark a candidate invalid. Duplicate
case IDs in JSONL, duplicate model names, missing tasks, and misaligned matrix
fields are rejected.

This script only transforms data. It requires no model configuration and does
not run inference or candidate code, read input CSVs, or create verifier
annotations, manifests, or caches.

Load the result with:

```python
from datasets import load_from_disk

dataset = load_from_disk("dataset/processed-prepbench")
```
