"""
Pipeline Fase 8 - Sicurezza IA: validazione input per prevenire prompt injection.
"""
import re
from typing import List, Optional, Tuple

# Pattern che suggeriscono tentativo di injection / override istruzioni (FractalNova)
INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(previous|above|all)\s+instructions?", re.I), "instruction_override"),
    (re.compile(r"forget\s+(everything|all)\s+above", re.I), "context_override"),
    (re.compile(r"system\s*:\s*", re.I), "system_prompt_leak"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "role_override"),
    (re.compile(r"<\s*script\s*", re.I), "xss"),
    (re.compile(r"javascript\s*:", re.I), "xss"),
    (re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<<\/SYS>>", re.I), "token_leak"),
    (re.compile(r"repeat\s+(forever|all)\s+above", re.I), "loop_injection"),
    (re.compile(r"disregard\s+(all|previous)", re.I), "instruction_override"),
    (re.compile(r"new\s+instruction\s*:", re.I), "instruction_override"),
    (re.compile(r"output\s+(only|just)\s+(the\s+)?(raw|internal)", re.I), "output_manipulation"),
]


def check_prompt_injection(prompt: str) -> Tuple[bool, Optional[str], List[str]]:
    """
    Controlla se il prompt contiene pattern sospetti.
    Restituisce (safe, None, []) se OK, altrimenti (False, reason, list_matched_types).
    """
    if not prompt or not prompt.strip():
        return True, None, []
    matched = []
    for pat, name in INJECTION_PATTERNS:
        if pat.search(prompt):
            matched.append(name)
    if matched:
        return False, f"Prompt injection sospetto: {', '.join(matched)}", matched
    return True, None, []


def sanitize_prompt_for_model(prompt: str, max_length: int = 50000) -> str:
    """Rimuove/neutralizza caratteri pericolosi e tronca."""
    if not prompt:
        return ""
    s = prompt.strip()
    # Rimuovi null bytes
    s = s.replace("\x00", "")
    if len(s) > max_length:
        s = s[:max_length]
    return s


def rate_limit_key(identifier: str) -> str:
    """Chiave per rate limiting chiamate modello (per user/IP)."""
    return f"model_calls:{identifier}"
