"""
Pipeline Fase 7 - Backup e Disaster Recovery: restore da SecureBackupManager.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from .dr_config import get_dr_config


def get_backup_dir(root: str = None) -> str:
    if root is None:
        root = os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root, "security", "secure_backups")


def list_available_restores(root: str = None) -> List[Dict]:
    """Elenco backup disponibili (da metadata SecureBackupManager)."""
    backup_dir = get_backup_dir(root)
    meta_path = os.path.join(backup_dir, "backup_metadata.json")
    if not os.path.isfile(meta_path):
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    out = []
    for filepath, entries in data.get("backups", {}).items():
        for entry in entries:
            out.append({
                "source_file": filepath,
                "backup_file": entry.get("backup_path"),
                "timestamp": entry.get("timestamp"),
                "checksum": entry.get("checksum"),
            })
    return sorted(out, key=lambda x: x.get("timestamp") or "", reverse=True)


def run_restore(
    target_file: str,
    backup_path: str,
    root: str = None,
    dry_run: bool = True,
) -> Dict[str, str]:
    """
    Ripristina file da backup. dry_run=True solo verifica senza scrivere.
    Restituisce dict con status, message.
    """
    root = root or os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
    if not os.path.isfile(backup_path):
        return {"status": "error", "message": f"Backup non trovato: {backup_path}"}
    full_target = os.path.join(root, target_file) if not os.path.isabs(target_file) else target_file
    if dry_run:
        return {"status": "dry_run", "message": f"Would restore {backup_path} -> {full_target}"}
    try:
        import shutil
        shutil.copy2(backup_path, full_target)
        return {"status": "ok", "message": f"Restored {target_file}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_dr_procedure(steps: List[str] = None, root: str = None) -> Dict:
    """Esegue procedura DR (runbook). steps = elenco passi da dr_config."""
    cfg = get_dr_config()
    steps = steps or cfg.get("restore_steps", [])
    root = root or os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
    results = []
    for step in steps:
        results.append({"step": step, "status": "documented", "note": "Eseguire manualmente"})
    return {
        "phase": "7_dr",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rto_hours": cfg.get("rto_hours"),
        "rpo_hours": cfg.get("rpo_hours"),
        "steps": results,
    }
