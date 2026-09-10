# Dataset processing

Use `qrank-verifier transform-dataset` to add semantic judgments to candidate
solutions. See [setup and label meanings](README.md) first.

## Run a pilot, then the full dataset

```bash
uv run qrank-verifier transform-dataset \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output dataset/verified-pilot \
  --split test \
  --limit 5
```

For a full run, omit `--limit` and choose a new output directory. This example
reuses successful pilot judgments:

```bash
uv run qrank-verifier transform-dataset \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output dataset/verified-full \
  --split test \
  --cache dataset/verified-pilot/verifier_cache.sqlite \
  --batch-size 8
```

Paths supplied on the command line are relative to the working directory.
The output must be outside the input dataset directory. A completed output
DatasetDict cannot be overwritten.

## Prepare input and context

Input can be a Hugging Face `DatasetDict` saved with `save_to_disk`, or a `.jsonl`
file with one question object per line. JSONL rows are loaded into `--split`.
Each row needs a unique, nonempty `question_id`, a `question`, and a `candidates`
list. Each candidate needs a `text` string; optional `invalid_mask=true` marks it
for skipping. Optional `evidence` and `context` supply task details.

For text-to-SQL, each row also requires an N-by-N `agreement` matrix for its N
candidates. Agreement among valid candidates must be symmetric and transitive.
The pipeline removes invalid/empty candidates, verifies the first valid member
of each equivalence cluster, and shares that result with the cluster's members.
It uses supplied agreement; it does not execute SQL to compute it.

Supply SQL schemas explicitly when needed:

```bash
uv run qrank-verifier transform-dataset \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output dataset/verified-sql \
  --task-type text-to-sql \
  --schemas path/to/schema.json
```

`--schemas` accepts a database-to-DDL JSON mapping (values may be strings or lists
of strings), or a BIRD tables JSON list. Database IDs are taken from
`metadata.db_id`, falling back to `group_id`. If schemas are loaded, every SQL
row must match a database ID. Include dialect and additional requirements in the
row's `context`; they are retained alongside the schema.

Without an explicit schema path, discovery searches the repository's
`dataset/base` for `schema.json`, then `dev_tables.json`, then extracted `.sqlite`
files. Multiple matching schema files require `--schemas`. Alternatively, use
`--db-root path/to/databases` without `--schemas` to read table/view definitions
from SQLite databases using read-only connections. If discovery finds no schemas
or databases, verification uses the supplied row context.

For PrepBench:

```bash
uv run qrank-verifier transform-dataset \
  --input dataset/raw/prepbench.jsonl \
  --output dataset/verified-prepbench \
  --task-type prepbench \
  --prepbench-root path/to/prepbench-inputs
```

The optional input root must contain `<question_id>/inputs/*.csv` for every
processed question. The verifier reads headers and up to five sample rows per
file; missing files under an explicit root are an error. Omit the flag if inputs
are unavailable and provide table schemas, samples, runtime dtypes, and available
imports/globals in row `context` where possible. CSV samples do not establish
runtime pandas dtypes. The candidate interface is
`solve(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]`, with filenames
as dictionary keys.

PrepBench rows may omit `agreement`: matching nonempty candidate `cluster_id`
values define equivalence, and candidates without IDs are verified individually.
The default `--task-type auto` checks row/metadata `task_type`, then a candidate
`def solve(tables...)` signature, otherwise assumes SQL. Use an explicit task
type when detection would be ambiguous. The local export also has a
[convenience script](../scripts/prepbench/README.md).

## Inspect the output

The output is a saved DatasetDict with original columns and candidate order
preserved. Only the selected split is annotated; with `--limit`, that split
contains only the selected prefix. Other splits are retained.

```python
import json
from pathlib import Path
from datasets import load_from_disk

output = Path("dataset/verified-full")
data = load_from_disk(str(output))
row = data["test"][0]
for candidate, annotation in zip(
    row["candidates"], row["semantic_verifier"], strict=True
):
    print(candidate["text"])
    print(annotation["status"], annotation["labels"]["overall"])
    print(annotation["reasons"]["overall"])

manifest = json.loads((output / "semantic_verifier.manifest.json").read_text())
print(manifest["n_failed_verifications"], manifest["errors"])
```

`semantic_verifier` has one entry per candidate, containing `cluster_id`,
`status`, `cache_key`, `labels`, and `reasons`. Status is `ok` for a fresh
judgment, `cached` for a reused judgment, `failed` after exhausted retries, or
`invalid` for an invalid/empty candidate. Output cluster IDs are strings derived
from representative candidate indices, not the original PrepBench IDs.

The manifest records model identity, prompt version, run options, question and
cluster counts, label counts, and validation errors. Counts of labels and failed
verifications are **per cluster**, not per candidate. `n_unverified_clusters=0`
does not mean every verification succeeded; check `n_failed_verifications` too.
A run can save successfully while containing failed judgments.

By default, successful responses are stored in `<output>/verifier_cache.sqlite`.
With `--cache`, the cache stays at the supplied path. The manifest describes the
run that created it: an older manifest's prompt version need not match the
current source in `src/qrank/verifier/prompts.py`.

## Configure and resume

| Option | Default | Use |
| --- | --- | --- |
| `--backend` | `transformers` | Registered model backend |
| `--model` | `Qwen/Qwen3-1.7B` | Model name or local path |
| `--device` | `auto` | Use `cpu`, `cuda`, or `cuda:0` explicitly |
| `--dtype` | `auto` | Also accepts `float32`, `float16`, `bfloat16` |
| `--temperature` | `0.7` | Set `0` for greedy decoding |
| `--max-tokens` | `700` | Maximum generated response tokens |
| `--revision` | `main` | Pin a model revision or commit |
| `--batch-size` | `8` | Candidate prompts per generation batch |
| `--retries` | `2` | Additional attempts after the first failure |
| `--retry-delay` | `1.0` | Initial retry delay in seconds, doubled on each retry |
| `--progress-every` | `10` | Question grouping for progress; `0` disables progress lines |

Model defaults come from [config.py](../src/qrank/config.py); flags override them.
The Transformers backend disables thinking and uses top-p `0.8` and top-k `20`
when sampling. Each prompt contains one representative candidate, even when
generation batches span multiple questions.

If inference is interrupted before a DatasetDict has been saved, rerun the
command to reuse successful cached judgments. After a completed save, use a new
`--output` and point `--cache` to the previous cache to retry failed judgments.
Only successes are cached. Keys include task content, context, prompt version,
system prompt, and backend identity; changing those inputs causes fresh work.

For GPU out-of-memory errors, reduce `--batch-size` and rerun. Runtime errors
propagate rather than becoming semantic failures. For repeated format failures,
inspect the failure reasons and consider increasing `--max-tokens`. Prompt plus
response must fit the model context window. Fresh sampled results may differ
with batching changes.

List all flags with `uv run qrank-verifier transform-dataset --help`.
