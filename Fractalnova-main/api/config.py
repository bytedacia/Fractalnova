"""Configurazione del servizio API (da ambiente / .env)."""
from __future__ import annotations

import os


def _origins() -> list[str]:
    raw = os.getenv("API_CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings:
    app_name: str = "FractalNova API"
    version: str = "0.1.0"

    # Auth / JWT
    secret_key: str = os.getenv("API_JWT_SECRET", "dev-secret-change-me")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("API_TOKEN_TTL_MIN", "60"))

    # Persistenza
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./fractalnova_api.db")

    # CORS + rate limit
    cors_origins: list[str] = _origins()
    rate_limit_per_min: int = int(os.getenv("API_RATE_LIMIT_PER_MIN", "60"))

    # Generazione
    max_pages: int = int(os.getenv("API_MAX_PAGES", "100000"))


settings = Settings()
