"""Optional local vLLM backend; no GPU dependencies load at import time."""
from __future__ import annotations

import math
import copy
import threading
from typing import Any, Sequence

from qrank.config import LLMConfig

from .base import BackendError, LLMBackend
from .registry import register_backend


@register_backend("vllm")
class VLLMBackend(LLMBackend):
    """Run native batched inference using one serialized offline engine."""

    def __init__(self, model: str, device: str, dtype: str, temperature: float,
                 max_tokens: int, revision: str = "main", *,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None, enable_prefix_caching: bool = True,
                 quantization: str | None = None, seed: int = 0):
        super().__init__(model)
        LLMConfig("vllm", model, device, dtype, temperature, max_tokens, revision)
        if device not in ("auto", "cuda"):
            raise BackendError("vllm requires device='cuda' or 'auto'; select GPUs with "
                               "CUDA_VISIBLE_DEVICES before starting the process")
        if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
            raise BackendError("tensor_parallel_size must be a positive integer")
        if (isinstance(gpu_memory_utilization, bool)
                or not isinstance(gpu_memory_utilization, (int, float))
                or not math.isfinite(gpu_memory_utilization)
                or not 0 < gpu_memory_utilization <= 1):
            raise BackendError("gpu_memory_utilization must be in (0, 1]")
        if max_model_len is not None and (
            type(max_model_len) is not int or max_model_len <= max_tokens
        ):
            raise BackendError("max_model_len must be an integer greater than max_tokens")
        if type(enable_prefix_caching) is not bool:
            raise BackendError("enable_prefix_caching must be a boolean")
        if type(seed) is not int or seed < 0:
            raise BackendError("seed must be a non-negative integer")
        if quantization is not None and (
            not isinstance(quantization, str) or not quantization.strip()
        ):
            raise BackendError("quantization must be a non-empty string or null")
        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendError("vLLM is unavailable; install the optional backend with "
                               "`uv sync --extra vllm` on Linux with a compatible GPU") from exc

        self.lock = threading.Lock()
        self.max_tokens = max_tokens
        engine_options = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "quantization": quantization, "seed": seed,
            "structured_outputs_config": {"backend": "xgrammar"},
        }
        if max_model_len is not None:
            engine_options["max_model_len"] = max_model_len
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens,
            top_p=0.8 if temperature > 0 else 1.0,
            top_k=20 if temperature > 0 else -1,
        )
        self.lm = LLM(
            model=model, revision=revision, tokenizer_revision=revision,
            dtype=dtype, generation_config="vllm", **engine_options,
        )
        self.tokenizer = self.lm.get_tokenizer()
        model_config = self.lm.llm_engine.model_config
        self.max_model_len = model_config.max_model_len
        self.config = {
            "revision": revision,
            "resolved_revision": getattr(model_config.hf_config, "_commit_hash", None),
            "device": device, "dtype": str(model_config.dtype),
            "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": False,
            "top_p": self.sampling_params.top_p, "top_k": self.sampling_params.top_k,
            "vllm_version": vllm.__version__, "generation_config": "vllm",
            **engine_options, "max_model_len": self.max_model_len,
        }

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), **self.config}

    def generate(self, *, question_id: str, cluster_id: str, system: str, user: str) -> str:
        return self.generate_batch([{"system": system, "user": user}])[0]

    def generate_batch(self, requests: Sequence[dict[str, Any]]) -> list[str]:
        if not requests:
            return []
        with self.lock:
            prompts = []
            sampling = []
            for request in requests:
                token_ids = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": request["system"]},
                     {"role": "user", "content": request["user"]}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
                if len(token_ids) + self.max_tokens > self.max_model_len:
                    raise BackendError("Prompt plus generation exceeds model context window")
                prompts.append({"prompt_token_ids": token_ids})
                params = self.sampling_params
                if "response_schema" in request:
                    from vllm.sampling_params import StructuredOutputsParams
                    params = copy.deepcopy(params)
                    params.structured_outputs = StructuredOutputsParams(
                        json=request["response_schema"],
                    )
                sampling.append(params)
            outputs = self.lm.generate(
                prompts, sampling if any("response_schema" in r for r in requests)
                else self.sampling_params, use_tqdm=False,
            )
            if len(outputs) != len(requests) or any(len(output.outputs) != 1 for output in outputs):
                raise BackendError("vLLM did not return one completion per request")
            for request, output in zip(requests, outputs, strict=True):
                completion = output.outputs[0]
                if "response_schema" in request and completion.finish_reason != "stop":
                    raise BackendError(
                        f"vLLM structured response did not finish: {completion.finish_reason}; "
                        f"question_id={request.get('question_id')}, cluster_id={request.get('cluster_id')}; "
                        f"raw_text={completion.text!r}"
                    )
            return [output.outputs[0].text.strip() for output in outputs]
