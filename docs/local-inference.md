# Local inference with vLLM

Both selector and verifier support `"backend": "vllm"` through the existing
`qrank.llms` interface. The native offline engine batches prompts, enables prefix
caching by default, and loads once per backend instance. No server is required.
Transformers remains available with the existing configs.

Install the optional dependency on Linux with a compatible NVIDIA GPU:

```bash
uv sync --extra vllm
uv run --no-sync qrank-selector \
  --config training_configs/selector_naive_qwen3_8b_prepbench_vllm.json
```

The extra pins vLLM 0.11.0 to match this project's PyTorch 2.8.0 and CUDA 12.8
stack. Upgrading vLLM requires checking its PyTorch/CUDA requirements together;
see the [vLLM installation guide](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html).
The backend imports vLLM only when constructed, so the extra is unnecessary for
Transformers users. Use `uv run --extra vllm` or `uv run --no-sync` after installing;
a plain `uv run` can sync away optional dependencies.

The supplied PrepBench preset uses the same model, sampling settings, token
budget, and dataset as the Transformers 8B preset, with a separate output
directory and `batch_size: 16`. Adjust paths for your dataset. It permits 32,768
tokens per prompt plus completion and reserves 85% of GPU memory for the engine.
These are starting settings, not measured optimal values.

For a verifier run, copy your verifier JSON, change `backend` to `vllm`, set
`device` to `cuda`, and optionally add:

```json
"backend_kwargs": {
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.85,
  "max_model_len": 32768,
  "enable_prefix_caching": true,
  "seed": 0
}
```

Add `"batch_size": 16` at the top level of the verifier JSON, then pass it to
`Verifier(config_path=...)` or `qrank-verifier transform-dataset --config PATH`.
An explicit API `batch_size` argument or CLI `--batch-size` overrides the file;
configs without the field default to 8. The supplied 8B verifier config uses 16.
`backend_kwargs` is optional and forwards backend-specific constructor options;
it cannot replace shared settings such as `model`, `dtype`, or `max_tokens`.
Remove vLLM-specific kwargs when switching back to Transformers.

| Setting | Default | Purpose |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | Spread model weights across this many visible GPUs |
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory budgeted for the engine |
| `max_model_len` | Model default | Total prompt plus completion context limit |
| `enable_prefix_caching` | true | Reuse KV states for identical prompt prefixes |
| `quantization` | null | vLLM quantization method; requires a compatible model/setup |
| `seed` | 0 | Engine random seed |

Use `device: "cuda"` or `"auto"`. Select physical GPUs before launching with
`CUDA_VISIBLE_DEVICES=0` (or `0,1` with `tensor_parallel_size: 2`); this backend
does not accept Transformers device maps such as `cuda:1` or support CPU inference.

Larger application batches give vLLM more requests to schedule concurrently.
Try 8, 16, and 32 and compare steady-state rows/sec after model initialization
and compilation. Prefix caching can help repeated system/context prefixes;
it does not skip decoding. See vLLM's
[offline generation API](https://docs.vllm.ai/en/v0.11.0/api/vllm/entrypoints/llm.html)
and [prefix caching documentation](https://docs.vllm.ai/en/v0.11.0/features/automatic_prefix_caching.html).
Calls on a backend instance are serialized; requests from separate API calls are
not combined. For an application with many concurrent callers, a separately
served asynchronous engine would require another integration.

Qwen thinking remains disabled. Positive temperatures use `top_p=0.8` and
`top_k=20`; zero temperature uses greedy decoding. Model generation-config
defaults are disabled so they cannot silently override the configured budget.
Prompts exceeding the context budget fail without truncation. Cache identities
include the backend, resolved model revision, vLLM version, and engine settings.
Outputs can differ across engines even with greedy decoding; compare selection
accuracy as well as speed before adopting a run configuration.

The backend contract and config wiring have CPU-only mocked tests. GPU throughput
and the supplied preset have not been benchmarked.
