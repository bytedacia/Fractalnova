"""
Pipeline Fase 6 - Monitoraggio: regole IDS (anomalie, brute force, integrità).
"""
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from security.central_logger import log_anomaly
    from security.alerting import alert_brute_force
except ImportError:
    from .central_logger import log_anomaly
    from .alerting import alert_brute_force


# Limite tentativi auth per IP in finestra (secondi)
AUTH_WINDOW_SEC = 600
AUTH_MAX_ATTEMPTS = 10

_auth_attempts: Dict[str, List[float]] = defaultdict(list)


def record_auth_attempt(ip: str, success: bool) -> Optional[str]:
    """
    Registra tentativo login. Se supera soglia restituisce messaggio e triggera alert.
    """
    now = time.time()
    key = ip or "unknown"
    _auth_attempts[key].append(now)
    # Mantieni solo tentativi nella finestra
    _auth_attempts[key] = [t for t in _auth_attempts[key] if now - t < AUTH_WINDOW_SEC]
    if len(_auth_attempts[key]) >= AUTH_MAX_ATTEMPTS:
        alert_brute_force(key, len(_auth_attempts[key]))
        log_anomaly(f"Brute force: {key} ({len(_auth_attempts[key])} tentativi)", "HIGH", {"ip": key})
        return "Troppi tentativi di accesso. Blocco temporaneo."
    return None


def check_path_tampering(expected_hashes: Dict[str, str], current_hashes: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Confronta hash attuali con baseline. Restituisce lista (path, motivo) per file alterati.
    """
    from security.alerting import send_alert
    violations = []
    for path, expected in expected_hashes.items():
        current = current_hashes.get(path)
        if current is None:
            violations.append((path, "file_missing"))
        elif current != expected:
            violations.append((path, "hash_mismatch"))
            send_alert("Integrity check failed", "HIGH", f"Hash mismatch: {path}", {"path": path})
    return violations


def get_anomaly_rules() -> Dict[str, dict]:
    """Regole configurabili per detection (estensibile)."""
    return {
        "auth_brute_force": {"window_sec": AUTH_WINDOW_SEC, "max_attempts": AUTH_MAX_ATTEMPTS},
        "integrity_baseline": {"description": "Confronto hash file critici con baseline"},
    }
