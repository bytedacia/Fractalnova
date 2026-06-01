"""
Pipeline Fase 6 - Monitoraggio: alert automatici su eventi sospetti.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# Integrazione con central_logger
try:
    from security.central_logger import log_security_event
except ImportError:
    from .central_logger import log_security_event


def _load_alert_config() -> Dict:
    path = os.path.join(os.path.dirname(__file__), "alert_config.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "enabled": True,
        "webhook_url": os.getenv("SECURITY_ALERT_WEBHOOK", ""),
        "email_enabled": False,
        "min_severity": "MEDIUM",
        "rate_limit_alerts_per_hour": 10,
    }


_alert_count: Dict[str, int] = {}
_alert_count_ts: Optional[datetime] = None


def _check_rate_limit() -> bool:
    """Max N alert per ora per non saturare."""
    global _alert_count_ts
    cfg = _load_alert_config()
    limit = cfg.get("rate_limit_alerts_per_hour", 10)
    now = datetime.utcnow()
    if _alert_count_ts is None or (now - _alert_count_ts).total_seconds() > 3600:
        _alert_count.clear()
        _alert_count_ts = now
    key = now.strftime("%Y%m%d%H")
    _alert_count[key] = _alert_count.get(key, 0) + 1
    return _alert_count[key] <= limit


def send_alert(
    title: str,
    severity: str = "MEDIUM",
    message: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Invia alert: log + eventuale webhook/email.
    severity: LOW | MEDIUM | HIGH | CRITICAL
    """
    cfg = _load_alert_config()
    if not cfg.get("enabled", True):
        return False
    min_sev = cfg.get("min_severity", "MEDIUM")
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if order.index(severity) < order.index(min_sev):
        return False
    if not _check_rate_limit():
        return False
    log_security_event(
        "anomaly", title, level=severity,
        details={"message": message, **(details or {})}, success=False,
    )
    webhook = cfg.get("webhook_url") or os.getenv("SECURITY_ALERT_WEBHOOK")
    if webhook and message:
        _send_webhook(webhook, title, severity, message, details)
    return True


def _send_webhook(url: str, title: str, severity: str, message: str, details: Optional[Dict]) -> None:
    """POST a webhook (Slack, Teams, etc.)."""
    try:
        import urllib.request
        body = json.dumps({
            "title": title,
            "severity": severity,
            "message": message,
            "details": details,
            "ts": datetime.utcnow().isoformat() + "Z",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            pass
    except Exception:
        pass


def alert_brute_force(ip: str, count: int) -> None:
    """Alert tentativi brute force."""
    send_alert(
        "Brute force attempt",
        severity="HIGH",
        message=f"Troppi tentativi da IP {ip}",
        details={"ip": ip, "attempts": count},
    )


def alert_integrity_failure(path: str, reason: str) -> None:
    """Alert fallimento integrità file."""
    send_alert(
        "Integrity check failed",
        severity="HIGH",
        message=reason,
        details={"path": path},
    )
