"""
Pipeline Fase 5 - Sicurezza codice: token anti-CSRF per form e API.
"""
import os
import hmac
import hashlib
import secrets
from typing import Optional


def _get_secret() -> bytes:
    return os.getenv("CSRF_SECRET", secrets.token_hex(32)).encode("utf-8")


def generate_csrf_token(session_id: Optional[str] = None) -> str:
    """Genera token CSRF (HMAC del session_id o random)."""
    if session_id:
        return hmac.new(_get_secret(), session_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return secrets.token_hex(32)


def validate_csrf_token(token: str, session_id: Optional[str] = None) -> bool:
    """Valida token. Se session_id fornito, verifica HMAC; altrimenti solo presenza/non vuoto."""
    if not token or not token.strip():
        return False
    if session_id:
        expected = generate_csrf_token(session_id)
        return hmac.compare_digest(token.strip(), expected)
    return len(token.strip()) >= 32  # accetta qualsiasi token lungo se non c'è sessione
