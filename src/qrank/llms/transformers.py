"""Transformers backend; heavy dependencies load only on construction."""
from __future__ import annotations

import threading
from typing import Any, Sequence

from .base import BackendError, LLMBackend
from .registry import register_backend


@register_backend("transformers")
class TransformersBackend(LLMBackend):
    """Load one local causal LM and serialize access to its generation state."""

    backend_name = "transformers"

    def __init__(self, model: str, device: str, dtype: str, temperature: float,
                 max_tokens: int, revision: str = "main"):
        super().__init__(model)
        if max_tokens <= 0 or temperature < 0:
            raise BackendError("max-tokens must be positive and temperature nonnegative")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, device_map=device,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        ).eval()
        self.config = {
            "revision": revision, "resolved_revision": getattr(self.lm.config, "_commit_hash", None),
            "device": device, "dtype": str(self.lm.dtype),
            "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": False, "top_p": 0.8, "top_k": 20,
            "transformers_version": transformers.__version__,
        }

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), **self.config}

    def generate(self, *, question_id: str, cluster_id: str, system: str, user: str) -> str:
        return self.generate_batch([{"system": system, "user": user}])[0]

    def generate_batch(self, requests: Sequence[dict[str, str]]) -> list[str]:
        with self.lock, self.torch.inference_mode():
            prompts = [self.tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system"]},
                 {"role": "user", "content": r["user"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for r in requests]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(self.lm.device)
            length = inputs["input_ids"].shape[-1]
            if length + self.max_tokens > self.lm.config.max_position_embeddings:
                raise BackendError("Prompt plus generation exceeds model context window")
            sampling = ({"temperature": self.temperature, "top_p": 0.8, "top_k": 20}
                        if self.temperature > 0 else {})
            output = self.lm.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id, **sampling,
            )
            return [text.strip() for text in self.tokenizer.batch_decode(
                output[:, length:], skip_special_tokens=True)]
