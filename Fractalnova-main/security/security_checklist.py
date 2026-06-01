"""
Pipeline Fase 9 - Training e cultura: checklist sicurezza (OWASP, runbook, contatti).
"""
from datetime import datetime
from typing import List, Dict, Any


def get_security_checklist() -> List[Dict[str, Any]]:
    """Checklist operativa per audit e training."""
    return [
        {"id": "1", "area": "Assessment", "item": "Inventario asset e dipendenze aggiornato", "done": False},
        {"id": "2", "area": "Assessment", "item": "Nessuna credenziale hardcoded (scan eseguito)", "done": False},
        {"id": "3", "area": "Hardening", "item": "CSP e security headers attivi", "done": False},
        {"id": "4", "area": "Hardening", "item": "Input sanitizzati (prompt, form)", "done": False},
        {"id": "5", "area": "Auth", "item": "API key / JWT configurati per ambiente prod", "done": False},
        {"id": "6", "area": "Auth", "item": "MFA per admin abilitata (se applicabile)", "done": False},
        {"id": "7", "area": "Dati", "item": "Backup cifrati e test restore eseguito", "done": False},
        {"id": "8", "area": "Dati", "item": "Log anonimizzati (GDPR)", "done": False},
        {"id": "9", "area": "Codice", "item": "SAST (Bandit) in CI e nessun finding critico aperto", "done": False},
        {"id": "10", "area": "Codice", "item": "Dependency scan (pip audit) eseguito", "done": False},
        {"id": "11", "area": "Monitoraggio", "item": "Log sicurezza centralizzati e retention configurata", "done": False},
        {"id": "12", "area": "Monitoraggio", "item": "Alert su brute force e integrità configurati", "done": False},
        {"id": "13", "area": "DR", "item": "RTO/RPO documentati e test restore mensile pianificato", "done": False},
        {"id": "14", "area": "IA", "item": "Prompt injection check e rate limit modello attivi", "done": False},
        {"id": "15", "area": "Continuous", "item": "Threat model e pentest pianificati", "done": False},
    ]


def checklist_report() -> Dict[str, Any]:
    """Report checklist per documentazione e KPI."""
    items = get_security_checklist()
    by_area = {}
    for i in items:
        a = i["area"]
        if a not in by_area:
            by_area[a] = {"total": 0, "done": 0}
        by_area[a]["total"] += 1
        if i.get("done"):
            by_area[a]["done"] += 1
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checklist": items,
        "by_area": by_area,
        "total": len(items),
        "completed": sum(1 for i in items if i.get("done")),
    }
