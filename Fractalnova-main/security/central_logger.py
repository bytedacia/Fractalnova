"""
Pipeline Fase 6 - Monitoraggio: logging centralizzato sicurezza (auth, modifiche, errori, anomalie).
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional


LOG_DIR = os.getenv("SECURITY_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "90"))


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _log_file(category: str) -> str:
    """File di log per categoria (es. auth, access, errors)."""
    return os.path.join(LOG_DIR, f"security_{category}.jsonl")


def log_security_event(
    category: str,
    event: str,
    level: str = "INFO",
    user_id: Optional[str] = None,
    ip: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True,
) -> None:
    """
    Scrive evento in log strutturato (JSONL).
    category: auth | access | data_change | error | anomaly
    """
    _ensure_log_dir()
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "event": event,
        "level": level,
        "success": success,
    }
    if user_id:
        entry["user_id"] = user_id
    if ip:
        entry["ip"] = ip
    if details:
        entry["details"] = details
    path = _log_file(category)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_auth(user_id: Optional[str], ip: Optional[str], success: bool, method: str = "api_key") -> None:
    """Log tentativo autenticazione."""
    log_security_event(
        "auth", "login", level="INFO" if success else "WARNING",
        user_id=user_id, ip=ip, details={"method": method}, success=success,
    )


def log_access(resource: str, user_id: Optional[str], ip: Optional[str], action: str = "read") -> None:
    """Log accesso a risorsa sensibile."""
    log_security_event(
        "access", action, user_id=user_id, ip=ip, details={"resource": resource},
    )


def log_data_change(resource: str, action: str, user_id: Optional[str], details: Optional[Dict] = None) -> None:
    """Log modifica dati."""
    log_security_event("data_change", action, user_id=user_id, details={"resource": resource, **(details or {})})


def log_error(message: str, component: str, details: Optional[Dict] = None) -> None:
    """Log errore sicurezza."""
    log_security_event("error", message, level="ERROR", details={"component": component, **(details or {})}, success=False)


def log_anomaly(description: str, severity: str = "MEDIUM", details: Optional[Dict] = None) -> None:
    """Log anomalia / evento sospetto."""
    log_security_event("anomaly", description, level=severity, details=details, success=False)
