"""
Pipeline Fase 7 - Backup e Disaster Recovery: RTO, RPO, passi restore.
"""
import os
import json
from typing import Any, Dict, List


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dr_config.json")

DEFAULT_DR = {
    "rto_hours": 4,
    "rpo_hours": 1,
    "backup_paths": [
        "security/secure_backups",
        "inference/configs",
        "app.py",
    ],
    "restore_steps": [
        "1. Verificare integrità backup (checksum)",
        "2. Ripristinare file critici da secure_backups",
        "3. Verificare configurazioni e variabili ambiente",
        "4. Riavviare servizi e eseguire smoke test",
        "5. Registrare evento in audit log",
    ],
    "offsite_location": "",
    "restore_test_schedule": "monthly",
}


def get_dr_config() -> Dict[str, Any]:
    """Carica configurazione DR (default se file assente)."""
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_DR.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_DR)


def save_dr_config(cfg: Dict[str, Any]) -> str:
    """Salva configurazione DR."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return CONFIG_PATH
