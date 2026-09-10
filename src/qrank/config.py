"""Shared LLM settings for the verifier API and CLI."""

LLM_BACKEND = "transformers"
LLM_OPTIONS = {
    "model": "Qwen/Qwen3-1.7B",
    "device": "auto",
    "dtype": "auto",
    "temperature": 0.7,
    "max_tokens": 700,
    "revision": "main",
}
