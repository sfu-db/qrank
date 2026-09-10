# Deployment in an application

Use the `Verifier` Python API to judge candidates at request time, without
creating a dataset. The module provides synchronous inference; it does not
include an HTTP server or a deployment CLI subcommand. See
[setup and label meanings](README.md) for installation and the result contract.

## Load once and verify requests

Create one verifier during application/worker startup and reuse it:

```python
from dataclasses import asdict
from qrank.verifier import Verifier

verifier = Verifier()

def verify_request(question: str, candidate: str, context: str,
                   evidence: str = "") -> dict:
    result = verifier.verify(
        question=question,
        candidate=candidate,
        context=context,
        evidence=evidence,
        task_type="text-to-sql",
    )
    return asdict(result)

response = verify_request(
    question="List all item IDs",
    candidate="SELECT id FROM items",
    context="SQLite. CREATE TABLE items (id INTEGER PRIMARY KEY);",
)
if response["status"] in {"ok", "cached"}:
    print(response["labels"]["overall"], response["reasons"]["overall"])
else:
    print("Verification unavailable:", response["reasons"]["overall"])
```

Run application scripts with `uv run python path/to/app.py` in the configured
environment. Importing `qrank.verifier` does not load model weights; constructing
`Verifier()` loads the configured backend. Initial construction may download
model/tokenizer files if they are not already available locally.

The API does not discover schema files or databases. Supply SQL DDL, dialect,
and relevant task details through `context`, and hints or definitions through
`evidence`. It judges the candidate as written without execution or repair.

## Verify several candidates

```python
candidates = ["SELECT id FROM items", "SELECT COUNT(*) FROM items"]
results = verifier.verify_candidates(
    question="List all item IDs",
    candidates=candidates,
    context="SQLite. CREATE TABLE items (id INTEGER PRIMARY KEY);",
    task_type="text-to-sql",
)
for candidate, result in zip(candidates, results, strict=True):
    print(candidate, result.status, result.labels["overall"])
```

Results preserve input order, and an empty sequence returns `[]`. Each candidate
is judged independently in its own prompt. Unlike dataset processing, this API
does not cluster candidates or filter invalid/empty candidates. Your application
supplies the candidates it wants judged; the verifier does not select a winner.

For Python/pandas tasks, use `task_type="prepbench"` and provide input/runtime
details explicitly:

```python
result = verifier.verify(
    question="Return items.csv with only its id column, preserving all rows.",
    candidate='''def solve(tables):
    return {"items.csv": tables["items.csv"][["id"]].copy()}
''',
    context=(
        'tables["items.csv"] is a pandas DataFrame with columns '
        'id (int64) and name (string). pandas is available as pd. '
        'The returned dictionary must contain the key "items.csv".'
    ),
    task_type="prepbench",
)
```

The interface is `solve(tables: dict[str, pd.DataFrame]) -> dict[str,
pd.DataFrame]`. Include output requirements, table schemas, representative
samples, actual runtime dtypes, and imports/globals as relevant. The API does
not read PrepBench CSV files. `task_type="auto"` detects a `def solve(tables...)`
signature among candidates and otherwise assumes SQL; an explicit type is useful
for requests with incomplete code.

## Configure the backend and cache

`Verifier()` reads `LLM_BACKEND` and `LLM_OPTIONS` from
[config.py](../src/qrank/config.py). Defaults are Transformers with
`Qwen/Qwen3-1.7B`, automatic device/dtype, temperature `0.7`, a `700`-token
response budget, and revision `main`. To configure an instance explicitly:

```python
from qrank.llms import load_backend
from qrank.verifier import SQLiteCache, Verifier

backend = load_backend(
    "transformers",
    model="Qwen/Qwen3-1.7B",
    device="auto",       # Or "cpu" / "cuda:0".
    dtype="auto",
    temperature=0,       # Greedy decoding.
    max_tokens=700,
    revision="main",    # Use a commit ID to pin the model version.
)
verifier = Verifier(
    backend=backend,
    cache=SQLiteCache("var/verifier_cache.sqlite"),
    retries=2,
    retry_delay=1.0,
    batch_size=8,
)
```

Passing a backend bypasses default backend construction. Caching is disabled
unless you pass a `SQLiteCache`; its path should be writable and persistent if
judgments should survive restarts. Successful responses are cached, failures are
not. Cache keys include question, evidence, built context, candidate, task type,
prompt version/system prompt, and backend identity. A matching cached response
returns `status="cached"` with zero attempts. Changing these inputs produces a
different key. Cached sampled judgments are reused rather than sampled again.

## Handle results and service execution

`VerificationResult` is a dataclass:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `cached`, or `failed` |
| `labels` | Perspective names mapped to `S`, `U`, `C`, or `NA` |
| `reasons` | Explanation for each perspective |
| `raw_text` | Model response for successful/cached results; empty on failure |
| `cache_key` | Identity of the judgment inputs and backend |
| `attempts` | Generation attempts for this call; zero for cache hits |

Always check `status` first. `failed` means verification was unavailable, with
`NA` in every label and failure details in the reasons. It is distinct from a
successful judgment of `C` or `U`. Expected generation/response-format failures
are retried up to `retries` additional times, with exponential delay. Runtime,
programming, and cache errors propagate to the application; handle them through
your service's normal error handling.

Calls block until inference and any retries finish. In an asynchronous server,
run them through the application's worker/executor mechanism. The built-in
Transformers backend serializes generation on each backend instance with a lock;
parallel requests sharing that instance do not run model generation concurrently.
`batch_size` batches candidates within a call, not requests across API calls.
Each process constructing its own backend loads its own model, so account for
model memory when choosing worker count.

Reduce `batch_size` if GPU memory is exhausted, and keep supplied context plus
`max_tokens` within the model context window. JSON responses are parsed and
validated after generation; generation is not constrained to a JSON grammar.
