# QRank semantic verifier

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.
The current workspace reuses its installed libraries through a virtual environment
with system site-packages enabled. Run from the project root:

```bash
uv run --no-sync qrank-verifier transform-dataset
```

This loads `Qwen/Qwen3-1.7B` once using Hugging Face Transformers, reads
`dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5`, and saves a Hugging Face DatasetDict
to `dataset/processed`. The default device is automatic; use `--device cpu` for
CPU inference or `--device cuda:0` for a specific GPU.

Generation batches up to 8 candidate prompts at a time, including across
questions. Set `--batch-size` to tune throughput and GPU memory use:

```bash
uv run --no-sync qrank-verifier transform-dataset \
  --batch-size 8 --device cuda:0 --dtype bfloat16
```

Each prompt still contains exactly one candidate. Cached responses bypass
generation, and only failed responses are retried. If CUDA runs out of memory,
reduce `--batch-size` and rerun; successful responses are already cached.
Batching changes sampling order, so fresh stochastic results can differ from
single-prompt runs. Memory use depends on prompt lengths and `--max-tokens`.

Relative paths are resolved from your shell's working directory. If you get
`Directory dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 not found`, first run
`cd /root/qrank` in this workspace. To run from another directory, use absolute
paths for the input, output, and schema discovery:

```bash
/root/qrank/.venv/bin/qrank-verifier transform-dataset \
  --input /root/qrank/dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output /root/qrank/dataset/processed \
  --schemas /root/qrank/dataset/base/bird/dev/dev_20240627/schema.json
```

The command automatically finds `schema.json` under the project's `dataset/base`
directory and uses the SQL definitions for each question's database. Discovery
works independently of the shell's working directory. If no `schema.json`
exists, it tries `dev_tables.json`, then reads table/view definitions from
extracted `.sqlite` files using read-only connections. Explicit paths are supported:

```bash
uv run --no-sync qrank-verifier transform-dataset \
  --schemas dataset/base/bird/dev/dev_20240627/schema.json
# Alternatively: --db-root path/to/dev_databases
```

Both database-to-DDL mappings (`schema.json`) and BIRD tables lists
(`dev_tables.json`) are supported. `--db-root` explicitly selects SQLite schema
discovery instead. The bundled schema covers all 11 BIRD development databases;
the database ZIP does not need extraction when using this JSON. If no schema
files or extracted databases exist, the command uses each row's context.

For a small pilot, use a separate output directory:

```bash
uv run --no-sync qrank-verifier transform-dataset \
  --limit 5 --output dataset/processed-pilot
```

Original columns, candidate ordering, and evaluation labels are preserved.
The new `semantic_verifier` column contains one annotation per candidate:
`cluster_id`, `status`, `cache_key`, `labels`, and `reasons`. Labels cover output,
constraints, ranking, relations, and overall semantics. Invalid/empty candidates
have `status=invalid` and NA labels. Invalid candidates are removed before
grouping by the agreement matrix; each remaining cluster is verified once and
its annotation is shared with its member candidates. Non-transitive or
asymmetric agreement is rejected.

Prompts contain only the question, evidence, schema, and one representative SQL.
Evaluation labels, votes, sources, and competing SQL are excluded. These are
model judgments, not correctness labels. Failed calls have `status=failed` and
NA labels, and are counted in the manifests.

The only input is a saved Hugging Face DatasetDict. Verification runs directly
on its selected split, without intermediate JSONL files. The output directory
also contains `semantic_verifier.manifest.json` and `verifier_cache.sqlite`.
Interrupted inference can be restarted with the same command: successful
responses are reused, and failures are retried. A completed DatasetDict is
protected from overwrite; choose a new `--output` to run again. Reuse a cache
explicitly with `--cache path/to/verifier_cache.sqlite`.

```python
from datasets import load_from_disk

data = load_from_disk("dataset/processed")
print(data["test"][0]["semantic_verifier"])
```

Qwen3 runs with thinking disabled, temperature 0.7, top-p 0.8, and top-k 20,
following the [model card](https://huggingface.co/Qwen/Qwen3-1.7B).
Use `--temperature 0` for greedy decoding, `--max-tokens` to change the response
budget, and `--revision` to pin a model commit. JSON is parsed and validated;
generation itself is not constrained to a JSON grammar.

Run the pipeline tests with `uv run --no-sync python -m unittest discover -s tests`.

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`, matching
the versions installed in this workspace: PyTorch 2.8.0+cu128, Transformers
4.57.6, Accelerate 1.15.0, and Datasets 4.8.5.
Use `uv add <package>` to add a dependency. The package is installed in editable
mode in `.venv`, so `uv run --no-sync python -m qrank.semantic_verifier` also works.
Linux and Windows use PyTorch 2.8 with CUDA 12.8 wheels; macOS uses PyPI wheels.
The `hf-transfer` dependency supports this workspace’s enabled Hugging Face
fast-download setting.

To recreate the local reuse setup when `.venv` does not exist:

```bash
uv venv --system-site-packages
uv pip install --no-deps -e .
```

`--no-sync` prevents uv from installing separate copies of the shared libraries.
This local reuse depends on the system environment; the lockfile does not enforce
its contents. For a self-contained environment in a fresh checkout, run
`uv sync --locked`, then use `uv run qrank-verifier` without `--no-sync`.
Do not repeat `uv venv` once `.venv` is set up.
