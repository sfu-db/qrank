# QRank semantic verifier

Usage guides: [dataset processing](docs/verifier-dataset.md) and
[deployment in an application](docs/verifier-deployment.md).

Applications can call the verifier directly, with no dataset setup:

```python
from qrank.verifier import Verifier

# Construct once and reuse across requests.
verifier = Verifier(config_path="training_configs/verifier_qwen3_1p7b.json")
result = verifier.verify(
    question="List all item IDs",
    candidate="SELECT id FROM items",
    context="CREATE TABLE items (id INTEGER PRIMARY KEY);",
    task_type="text-to-sql",
    sql_dialect="sqlite",
    perspective="output_semantics",
)
print(result.labels["output_semantics"], result.reasons["output_semantics"])
```

`verify_candidates(question, candidates, ...)` accepts a sequence of candidate
strings for the same task and returns results in input order. SQL first extracts
grounded requirements without candidates, then judges local block contributions
and individual requirement coverage. Omit `perspective` to request all four
perspectives; there is no overall verdict. SQL checking prompts include raw SQL,
a deterministic AST summary, and the shared contract. Identical candidates share judgments. Candidates are
not executed. Both methods accept `context`, `evidence`, `sql_dialect`, and
`task_type` (`auto`, `text-to-sql`, or `prepbench`).
For PrepBench, supply table schemas, samples, and runtime details in `context`;
the application API does not discover files. An empty candidate sequence returns
an empty list.

Model and generation settings live in
[training_configs/verifier_qwen3_1p7b.json](training_configs/verifier_qwen3_1p7b.json).
Copy or edit that JSON and pass its path with `Verifier(config_path=...)` or the
CLI's required `--config` option. `src/qrank/config.py` contains only the validated
`LLMConfig` and `VerifierConfig` data models. All seven model settings are required in the JSON; no model
preset is selected implicitly. Relative config paths resolve from the working
directory. CLI model flags override the file's values.

`Verifier` loads the backend once at construction; importing the module does not
load model weights. You can still pass `Verifier(backend)` for a custom or
already-loaded backend. Supply either a backend or a config path.

Set `batch_size` in the verifier JSON (8 when omitted). An explicit
`Verifier(..., batch_size=16)` or CLI `--batch-size 16` overrides the file.
Errors propagate immediately. There are no retry or fallback options.
Caching is disabled by default; pass `cache=SQLiteCache("verifier.sqlite")`
(imported from `qrank.verifier`) to persist and reuse successful judgments.
`VerificationResult` includes root labels/reasons, raw responses keyed by check ID,
status, a bundle cache key, attempt count, the shared `requirements` contract,
`block_results`, and separate `requirement_results`. See
[grounded weak signals](docs/verifier-block-signals.md) for evidence and dependency
metadata. Labels are `S` (satisfied), `U` (unresolved), `C`
(contradicted), or `NA` (no applicable local responsibility). Errors raise;
there is no `failed` result or NA error placeholder. JSON is parsed directly,
with no response repair. Calls are synchronous;
applications manage backend concurrency. The CLI and this API share the same
verification engine. Existing `verify_one`/`verify_many` functions remain
available for callers using `QuestionInput` and `ClusterInput`.

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.
The current workspace reuses its installed libraries through a virtual environment
with system site-packages enabled. Run from the project root:

```bash
uv run --no-sync qrank-verifier transform-dataset --config training_configs/verifier_qwen3_1p7b.json
```

This loads `Qwen/Qwen3-1.7B` once using Hugging Face Transformers, reads
`dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5`, and saves a Hugging Face DatasetDict
to `dataset/processed`. The default device is automatic; use `--device cpu` for
CPU inference or `--device cuda:0` for a specific GPU.

Generation batches candidate prompts using the config's `batch_size`, including
across questions. Use `--batch-size` to override it for a run:

```bash
uv run --no-sync qrank-verifier transform-dataset --config training_configs/verifier_qwen3_1p7b.json \
  --batch-size 8 --device cuda:0 --dtype bfloat16
```

Each prompt contains one candidate and one perspective rubric. Cached perspective
judgments bypass generation. If CUDA runs out of memory,
reduce `--batch-size` and rerun; successful responses are already cached.
Batching changes sampling order, so fresh stochastic results can differ from
single-prompt runs. Memory use depends on prompt lengths and `--max-tokens`.

Relative paths are resolved from your shell's working directory. If you get
`Directory dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 not found`, first run
`cd /root/qrank` in this workspace. To run from another directory, use absolute
paths for the input, output, and schema discovery:

```bash
/root/qrank/.venv/bin/qrank-verifier transform-dataset --config /root/qrank/training_configs/verifier_qwen3_1p7b.json \
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
uv run --no-sync qrank-verifier transform-dataset --config training_configs/verifier_qwen3_1p7b.json \
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
`uv run --no-sync python scripts/prepbench/transform.py --config training_configs/verifier_qwen3_1p7b.json`.
The [PrepBench script guide](scripts/prepbench/README.md) covers pilot runs,
input CSV context, and output/cache options.

```bash
uv run --no-sync qrank-verifier transform-dataset --config training_configs/verifier_qwen3_1p7b.json \
  --input dataset/raw/prepbench.jsonl \
  --output dataset/prepbench-verified --task-type prepbench
```

SQL and PrepBench share four semantic perspectives and S/U/C/NA meanings.
PrepBench retains code-level label/reason checks; SQL additionally uses grounded
requirements, block targets, and structured evidence. Per-question context and
candidate language labels appear in the user prompt.
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

The verifier uses `sql.py` for deterministic SQL summaries, `prompts.py` for
extraction and perspective rubrics, `responses.py` for response/evidence validation, and
`workflow.py` for batching and caching. `pipeline.py` prepares and annotates
datasets; `cli.py` and `api.py` expose the workflow. v13 check cache keys include
the prompt, fixed requirements, raw SQL/AST, parser version/dialect, context,
response schema, and backend identity. Extraction has a separate versioned key
that excludes candidates; old judgments are not reused.

vLLM constrains responses using xgrammar explicitly, with no decoder fallback.
Transformers/custom text backends receive the schema in the prompt without
constrained decoding. Malformed responses raise directly in either case.

Model backends live in `qrank.llms`. The verifier selects one with
`--backend transformers` or `--backend vllm`, or reads it from the config.
For native batched GPU inference, see [the vLLM setup and tuning guide](docs/local-inference.md).
`load_backend` uses `importlib` to import
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
uv run --no-sync qrank-verifier transform-dataset --config training_configs/verifier_qwen3_1p7b.json \
  --limit 5 --output dataset/processed-pilot
```

Original columns, candidate ordering, and evaluation labels are preserved.
The `semantic_verifier` column contains `labels`, `reasons`, `cache_key`, `status`,
`cluster_id`, and `agreement_cluster_id`. Only requested perspectives appear.
Invalid/empty candidates are explicitly skipped with `status=invalid` and null
labels. Distinct candidate texts within an agreement cluster are verified
separately; only identical stripped texts share a judgment. The original
agreement matrix is preserved and must remain symmetric and transitive.

Prompts exclude evaluation labels, votes, sources, and competing candidates.
These remain model judgments, not correctness labels. A wrong population filter
can be constraint C while a correct aggregate expression remains output S.
The AST describes candidate operations; it does not determine applicability.

Inputs may be a saved DatasetDict or JSONL. The output directory contains
`semantic_verifier.manifest.json`, `verifier_cache.sqlite`, and
`verifier_trace.jsonl`. The trace records requests before inference and raw
completions before parsing. A generation or parsing error raises immediately;
no completed dataset or success manifest is published. Fix the error, then rerun
the command to reuse successful perspective judgments. A completed
DatasetDict is protected from overwrite; use a new output directory to rerun it.


```python
from datasets import load_from_disk

data = load_from_disk("dataset/processed")
print(data["test"][0]["semantic_verifier"])
```

Qwen3 runs with thinking disabled, temperature 0.7, top-p 0.8, and top-k 20,
following the [model card](https://huggingface.co/Qwen/Qwen3-1.7B).
Use `--temperature 0` for greedy decoding, `--max-tokens` to change the response
budget, and `--revision` to pin a model commit. JSON is parsed directly;
vLLM additionally constrains decoding to the response schema.

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
