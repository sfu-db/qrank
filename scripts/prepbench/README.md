# Transform PrepBench

From the project root, annotate the local export with the configured verifier:

```bash
uv run --no-sync python scripts/prepbench/transform.py
```

Defaults: input `dataset/raw/prepbench.jsonl`, output
`dataset/processed-prepbench`, split `test`, task type `prepbench`.
The model and generation defaults come from `src/qrank/config.py`.

For a small pilot:

```bash
uv run --no-sync python scripts/prepbench/transform.py \
  --limit 5 --output dataset/processed-prepbench-pilot
```

All shared `transform-dataset` flags are available (`--help`), including
`--input`, `--output`, `--batch-size`, `--model`, and `--cache`.
Default paths work from any working directory; explicitly supplied relative
paths resolve from your working directory.

If input CSVs are available, pass `--prepbench-root PATH` with the layout
`PATH/<question_id>/inputs/*.csv` to include headers and five sample rows.
The bundled JSONL provides questions and candidates, without table context;
judgments may be unresolved when they depend on missing schemas or data.

The shared pipeline preserves source fields and candidate ordering, verifies
one representative per supplied equivalence cluster, and adds aligned
`semantic_verifier` annotations. It does not execute candidate code.
Output is a Hugging Face DatasetDict, with a verifier manifest and SQLite cache.
Interrupted inference can reuse the same output directory and cache. Completed
datasets cannot be overwritten; choose a new output and optionally reuse the
old cache with `--cache`.
