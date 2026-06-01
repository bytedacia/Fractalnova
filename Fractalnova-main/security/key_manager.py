"""
Pipeline Fase 4 - Protezione dati: gestione sicura chiavi (env, futuro HSM/KMS).
"""
import os
import base64
from typing import Optional

# Nomi variabili ambiente per chiavi (non esporre in log)
KEY_ENV_NAMES = (
    "API_SECRET",
    "SECURITY_ENC_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "SMTP_PASS",
    "DB_PASSWORD",
    "ENCRYPTION_KEY",
)


def get_secret(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """Legge secret da ambiente. Non loggare il valore."""
    return os.getenv(env_var, default)


def get_encryption_key(length: int = 32) -> bytes:
    """
    Restituisce chiave per cifratura. Da SECURITY_ENC_KEY (base64) o genera da env.
    In produzione preferire HSM/KMS.
    """
    raw = os.getenv("SECURITY_ENC_KEY")
    if raw:
        try:
            return base64.b64decode(raw)
        except Exception:
            pass
        try:
            return raw.encode("utf-8")[:length].ljust(length, b"\0")
        except Exception:
            pass
    # Fallback: genera da env deterministico (solo dev)
    import hashlib
    seed = os.getenv("SECURITY_KEY_SEED", "change-me-in-production")
    return hashlib.sha256(seed.encode()).digest()[:length]


def mask_secret(s: Optional[str], visible: int = 0) -> str:
    """Maschera secret per log (es. 'ab***')."""
    if not s:
        return "<unset>"
    if len(s) <= visible:
        return "*" * 3
    return s[:visible] + "*" * (len(s) - visible)
