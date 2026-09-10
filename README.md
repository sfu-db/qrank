# QRank semantic verifier

Applications can call the verifier directly, with no dataset setup:

```python
from qrank.verifier import Verifier

# Construct once and reuse across requests.
verifier = Verifier()
result = verifier.verify(
    question="List all item IDs",
    candidate="SELECT id FROM items",
    context="CREATE TABLE items (id INTEGER PRIMARY KEY);",
    task_type="text-to-sql",
)
if result.status in {"ok", "cached"}:
    print(result.labels["overall"], result.reasons["overall"])
else:
    print("Verification unavailable", result.reasons["overall"])
```

`verify_candidates(question, candidates, ...)` accepts a sequence of candidate
strings for the same task and returns results in input order. Each prompt contains
one candidate, with no clustering or candidate execution. Both methods accept
`context`, `evidence`, and `task_type` (`auto`, `text-to-sql`, or `prepbench`).
For PrepBench, supply table schemas, samples, and runtime details in `context`;
the application API does not discover files. An empty candidate sequence returns
an empty list.

Set `LLM_BACKEND` and `LLM_OPTIONS` in `src/qrank/config.py` to configure the model
and generation settings. `Verifier()` loads that backend once at construction;
importing the module does not load model weights. The CLI uses the same defaults,
with command-line flags taking precedence. You can still pass `Verifier(backend)`
for a custom or already-loaded backend.

Configure `Verifier(retries=2, retry_delay=1.0, batch_size=8)` as needed.
Caching is disabled by default; pass `cache=SQLiteCache("verifier.sqlite")`
(imported from `qrank.verifier`) to persist and reuse successful judgments.
`VerificationResult` includes labels, reasons, raw response text, status, cache
key, and attempt count. Labels are `S` (satisfied), `U` (unresolved), `C`
(contradicted), or `NA` (not applicable). Exhausted generation/format retries
return `status="failed"` with NA labels, which are not semantic verdicts.
Runtime, programming, and cache errors propagate. Calls are synchronous;
applications manage backend concurrency. The CLI and this API share the same
verification engine. Existing `verify_one`/`verify_many` functions remain
available for callers using `QuestionInput` and `ClusterInput`.

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

PrepBench is supported through `qrank.verifier.build_context`, which builds task
context for either dataset type. `--task-type auto` (the default) uses an explicit
`task_type` row/metadata field or a candidate `def solve(tables...)` signature;
otherwise it assumes text-to-SQL. Override with `--task-type prepbench` for
rows without that signature, including empty candidate sets.

For the local PrepBench export, run
`uv run --no-sync python scripts/prepbench/transform.py`.
The [PrepBench script guide](scripts/prepbench/README.md) covers pilot runs,
input CSV context, and output/cache options.

```bash
uv run --no-sync qrank-verifier transform-dataset \
  --input dataset/raw/prepbench.jsonl \
  --output dataset/prepbench-verified --task-type prepbench
```

SQL and PrepBench share the system prompt’s verdict rules, five semantic
perspectives, and output contract. Each adds its own background section describing
the task and candidate interface. Per-question context and candidate language
labels appear in the user prompt.
Raw rows use candidate `cluster_id` values when no agreement matrix is present;
candidates without IDs are verified individually. Missing `invalid_mask` means
no invalidity was supplied; empty candidates are still skipped. This step does
not execute code or establish runtime validity.

Context includes the task, evidence, supplied row context, and the `solve(tables)`
interface. Optional `--prepbench-root PATH` reads CSV headers and up to five rows
from `PATH/<question_id>/inputs/*.csv`. This is the supported input layout;
output/reference files outside `inputs` are never read. Missing input files under
an explicitly supplied root are an error. Without input context, the prompt
states that schemas, samples, and runtime dtypes are unavailable and requests
unresolved judgments where needed. CSV values do not imply runtime pandas dtypes.
SQL rows retain both database schemas and any additional row context.

The standalone `qrank.verifier` package owns the complete workflow:
`context.py` builds dataset context and discovers SQL schemas; `prompts.py`
defines the shared semantic contract, dataset backgrounds, and prompt builders; `models.py` defines shared inputs and results;
`qrank.llms` handles model generation, `cache.py` stores verifier responses; `pipeline.py` handles clustering,
verification, and annotation; and `cli.py` provides the command-line interface. Cache
keys include task type and system prompt. Prompt version v4 invalidates older
cached judgments.

Model backends live in `qrank.llms`. The verifier selects one with
`--backend transformers` (the default). `load_backend` uses `importlib` to import
the selected built-in module, whose `@register_backend` decorator registers its
implementation. Torch and Transformers load only when the backend is constructed.

```python
from qrank.llms import LLMBackend, load_backend, register_backend

@register_backend("custom")
class CustomBackend(LLMBackend):
    def generate(self, *, question_id, cluster_id, system, user):
        ...  # Return model response text.

backend = load_backend("custom", model="my-model")
```

Import a custom backend module before loading its registered name. New built-ins
also add their module path to `llms/registry.py`. The base class supplies sequential
batch generation; implementations may override it for native batching. CLI
backends accept the model and generation options passed by `cli.make_backend`.

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

Prompts contain only the question, evidence, task context, and one representative candidate.
Evaluation labels, votes, sources, and competing SQL are excluded. These are
model judgments, not correctness labels. Failed calls have `status=failed` and
NA labels, and are counted in the manifests.

Inputs may be a saved Hugging Face DatasetDict or a JSONL file (loaded into the selected split). Verification runs directly
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
mode in `.venv`, so `uv run --no-sync python -m qrank.verifier` also works.
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
