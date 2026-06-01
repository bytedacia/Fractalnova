"""
Pipeline Fase 2 - Hardening: sanitizzazione rigorosa input utente (XSS, injection, length).
"""
import re
import html
from typing import Optional

# Lunghezza massima campi tipici (caratteri)
MAX_PROMPT_LENGTH = 50_000
MAX_TITLE_LENGTH = 500
MAX_GENERIC_FIELD = 10_000

# Pattern per prompt injection (bloccare o segnalare)
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|all)\s+instructions?", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*script\s*", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"on\w+\s*=\s*['\"]", re.I),  # onclick=, onerror=, etc.
]


def sanitize_text(value: str, max_length: int = MAX_GENERIC_FIELD, allow_html: bool = False) -> str:
    """Escape HTML e tronca a max_length. Se allow_html=False rimuove tag."""
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if not allow_html:
        s = html.escape(s)
    if len(s) > max_length:
        s = s[:max_length]
    return s


def sanitize_prompt(prompt: str) -> str:
    """Sanitizza prompt utente: lunghezza e escape."""
    return sanitize_text(prompt, max_length=MAX_PROMPT_LENGTH)


def sanitize_title(title: str) -> str:
    """Sanitizza titolo (es. libro)."""
    return sanitize_text(title, max_length=MAX_TITLE_LENGTH)


def check_prompt_injection(text: str) -> Optional[str]:
    """
    Controlla pattern sospetti di prompt injection.
    Restituisce descrizione del pattern trovato o None se OK.
    """
    if not text:
        return None
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return f"Pattern sospetto: {pat.pattern[:50]}..."
    return None


def validate_and_sanitize_input(
    prompt: str = "",
    title: str = "",
    **kwargs
) -> tuple:
    """
    Valida e sanitizza input comuni. Restituisce (dict_sanitized, list_errors).
    """
    errors = []
    out = {}
    if prompt is not None:
        inj = check_prompt_injection(prompt)
        if inj:
            errors.append(inj)
        out["prompt"] = sanitize_prompt(prompt)
    if title is not None:
        out["title"] = sanitize_title(title)
    for k, v in kwargs.items():
        if isinstance(v, str):
            out[k] = sanitize_text(v, MAX_GENERIC_FIELD)
    return out, errors
