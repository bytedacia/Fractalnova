"""
Pipeline Fase 8 - Sicurezza IA: rate limiting sulle chiamate al modello (per IP/user).
"""
import os
import time
from collections import defaultdict
from typing import Optional

# Default: 30 richieste ogni 600 secondi (10 min) per chiave
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))
RATE_WINDOW_SEC = int(os.getenv("RATE_WINDOW_SEC", "600"))

_calls: defaultdict = defaultdict(list)


def _trim(key: str) -> None:
    now = time.time()
    _calls[key] = [t for t in _calls[key] if now - t < RATE_WINDOW_SEC]


def check_model_rate_limit(identifier: str) -> tuple:
    """
    Verifica se la chiave (IP o user_id) può effettuare una chiamata al modello.
    Restituisce (allowed: bool, remaining: int, retry_after_sec: Optional[int]).
    """
    key = f"model:{identifier}"
    _trim(key)
    if len(_calls[key]) >= RATE_LIMIT:
        oldest = min(_calls[key])
        retry_after = int(RATE_WINDOW_SEC - (time.time() - oldest))
        return False, 0, max(1, retry_after)
    return True, RATE_LIMIT - len(_calls[key]) - 1, None


def record_model_call(identifier: str) -> None:
    """Registra una chiamata al modello per la chiave."""
    key = f"model:{identifier}"
    _trim(key)
    _calls[key].append(time.time())
