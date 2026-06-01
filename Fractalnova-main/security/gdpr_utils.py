"""
Pipeline Fase 4 - Protezione dati: GDPR (diritto all'oblio, portabilità), anonimizzazione log.
"""
import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def anonymize_string(s: str, keep_prefix: int = 2, char: str = "*") -> str:
    """Anonimizza stringa lasciando solo prefisso (es. email: ab***@***)."""
    if not s or len(s) <= keep_prefix:
        return char * 3
    if "@" in s:
        local, domain = s.split("@", 1)
        local_a = local[:keep_prefix] + char * max(0, len(local) - keep_prefix)
        domain_a = char * min(3, len(domain)) + domain[domain.find("."):] if "." in domain else char * 3
        return f"{local_a}@{domain_a}"
    return s[:keep_prefix] + char * max(0, len(s) - keep_prefix)


def anonymize_log_line(line: str, email_pattern: Optional[re.Pattern] = None) -> str:
    """Sostituisce email e token nei log con versione anonimizzata."""
    if email_pattern is None:
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    return email_pattern.sub(lambda m: anonymize_string(m.group(0)), line)


def export_user_data_structure(user_id: str, data_sources: Dict[str, Any]) -> dict:
    """
    Struttura dati per portabilità (GDPR Art. 20).
    data_sources: dict con chiavi (es. profile, books, logs) e dati già caricati.
    """
    return {
        "export_date": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "data": data_sources,
        "format": "JSON",
    }


def delete_user_data_instructions() -> List[str]:
    """Istruzioni per diritto all'oblio (da eseguire manualmente o via job)."""
    return [
        "Rimuovere record utente da DB/app storage",
        "Rimuovere o anonimizzare log che referenziano user_id",
        "Rimuovere backup che contengono dati utente (o anonimizzare)",
        "Registrare data e scope della cancellazione in audit log",
    ]


def data_minimization_check(requested_fields: List[str], allowed_fields: List[str]) -> List[str]:
    """Restituisce solo i campi richiesti che sono anche consentiti (data minimization)."""
    return [f for f in requested_fields if f in allowed_fields]


def retention_policy_default_days() -> int:
    """Giorni di retention log (configurabile)."""
    return int(os.getenv("LOG_RETENTION_DAYS", "90"))
