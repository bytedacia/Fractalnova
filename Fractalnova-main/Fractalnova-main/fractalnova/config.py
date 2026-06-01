"""Configurazione centralizzata di FractalNova (letta dall'ambiente / .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # API
    google_api_key: str | None = None
    google_cse_api_key: str | None = None
    google_cse_cx: str | None = None
    hf_token: str | None = None

    # Percorsi modelli
    deepseek_path: str = "models/DeepSeek-V3"
    deepseek_config: str = "inference/configs/config_7B.json"
    qwen_path: str = "models/Qwen3-8B"
    llama_path: str = "models/Meta-Llama-3-8B-Instruct"
    gemma_path: str = "models/Gemma-7B"
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"

    # Comportamento
    download_models: bool = False
    allow_trust_remote_code: bool = False
    enable_auto_outreach: bool = False
    deepseek_max_new_tokens: int = 4096

    # Autore
    author_name: str = ""
    author_email: str = ""

    # SMTP
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_pass: str | None = None
    smtp_from: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 7860

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_api_key=os.getenv("GOOGLE_CSE_API_KEY"),
            google_cse_cx=os.getenv("GOOGLE_CSE_CX"),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
            deepseek_path=os.getenv("DEEPSEEK_LOCAL_PATH", "models/DeepSeek-V3"),
            deepseek_config=os.getenv("DEEPSEEK_CONFIG_PATH", "inference/configs/config_7B.json"),
            qwen_path=os.getenv("QWEN_LOCAL_MODEL_PATH", "models/Qwen3-8B"),
            llama_path=os.getenv("LLAMA_LOCAL_MODEL_PATH", "models/Meta-Llama-3-8B-Instruct"),
            gemma_path=os.getenv("GEMMA_LOCAL_MODEL_PATH", "models/Gemma-7B"),
            flux_model_id=os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev"),
            download_models=_flag("FRACTALNOVA_DOWNLOAD_MODELS"),
            allow_trust_remote_code=_flag("ALLOW_TRUST_REMOTE_CODE"),
            enable_auto_outreach=_flag("ENABLE_AUTO_OUTREACH"),
            deepseek_max_new_tokens=int(os.getenv("DEEPSEEK_MAX_NEW_TOKENS", "4096")),
            author_name=os.getenv("AUTHOR_NAME", ""),
            author_email=os.getenv("AUTHOR_EMAIL", ""),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_pass=os.getenv("SMTP_PASS"),
            smtp_from=os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "") or ""),
            host=os.getenv("FRACTALNOVA_HOST", "0.0.0.0"),
            port=int(os.getenv("FRACTALNOVA_PORT", "7860")),
        )
