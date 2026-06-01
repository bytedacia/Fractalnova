"""Client per il tier modello (endpoint OpenAI-compatible, es. vLLM).

Config via ambiente:
  MODEL_SERVER_URL       es. http://fractalnova-vllm:8001/v1
  MODEL_SERVER_NAME      nome del modello servito (default: fractalnova-pro)
  MODEL_SERVER_API_KEY   opzionale
"""
from __future__ import annotations

import os
from typing import Dict, List

from fractalnova.logging_config import get_logger

log = get_logger("fractalnova.serving")


def server_url() -> str | None:
    return os.getenv("MODEL_SERVER_URL")


def is_configured() -> bool:
    return bool(server_url())


def chat(messages: List[Dict[str, str]], temperature: float = 0.8,
         max_tokens: int = 32768, timeout: float = 600.0) -> str:
    """Chiamata chat/completions OpenAI-compatible. Solleva se non configurato."""
    url = server_url()
    if not url:
        raise RuntimeError("MODEL_SERVER_URL non configurato")
    import httpx

    headers = {}
    api_key = os.getenv("MODEL_SERVER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": os.getenv("MODEL_SERVER_NAME", "fractalnova-pro"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
