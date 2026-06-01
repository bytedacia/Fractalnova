"""
Pipeline Fase 10 - Continuous improvement: template threat model (STRIDE / asset-rischio).
"""
from datetime import datetime
from typing import Dict, List, Any


def get_threat_model_template() -> Dict[str, Any]:
    """Template per threat modeling trimestrale - FractalNova."""
    return {
        "version": "1.0",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "project": "FractalNova",
        "assets": [
            {"id": "A1", "name": "Modelli IA", "classification": "high", "description": "Pesi e config modelli"},
            {"id": "A2", "name": "Dati utente / libri", "classification": "high", "description": "Contenuti generati e metadati"},
            {"id": "A3", "name": "Credenziali e API key", "classification": "critical", "description": "Env e secrets"},
            {"id": "A4", "name": "Codice applicazione", "classification": "high", "description": "app.py, inference, security"},
        ],
        "threats": [
            {"id": "T1", "stride": "S", "description": "Spoofing: API key rubata o replay", "asset": "A3", "mitigation": "API key + rate limit, rotazione"},
            {"id": "T2", "stride": "T", "description": "Tampering: modifica codice o modelli", "asset": "A4,A1", "mitigation": "Integrità file, backup, guard"},
            {"id": "T3", "stride": "R", "description": "Repudiation: azioni non tracciate", "asset": "A2", "mitigation": "Audit log, central_logger"},
            {"id": "T4", "stride": "I", "description": "Information disclosure: prompt injection / leak", "asset": "A1,A2", "mitigation": "prompt_guard, output_sanitizer"},
            {"id": "T5", "stride": "D", "description": "DoS: rate limit superato o risorse esaurite", "asset": "A1", "mitigation": "model_rate_limiter, RATE_LIMIT"},
            {"id": "T6", "stride": "E", "description": "Elevation of privilege: accesso admin non autorizzato", "asset": "A3,A4", "mitigation": "RBAC, MFA admin"},
        ],
        "next_review": "Trimestrale",
    }


def threat_model_report() -> Dict[str, Any]:
    """Report threat model per documentazione."""
    return get_threat_model_template()
