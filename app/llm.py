from __future__ import annotations

import os

from langchain_ollama import OllamaLLM


def get_llm(model: str | None = None) -> OllamaLLM:
    return OllamaLLM(model=model or os.getenv("OLLAMA_MODEL", "mistral"))
