"""
Pipeline Fase 1 - Assessment: inventario asset critici, dipendenze, flusso dati.
FractalNova - Sistema IA Gestione Libri.
"""
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def get_project_root() -> str:
    return os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))


def inventory_critical_assets(root: str) -> Dict[str, Any]:
    """Mappatura asset critici: codice, modelli, config, dati."""
    assets = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project_root": root,
        "critical_paths": [],
        "config_files": [],
        "model_paths": [],
        "data_flows": [],
    }
    critical_dirs = [
        "inference",
        "security",
        "templates",
        "scripts",
    ]
    for d in critical_dirs:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            assets["critical_paths"].append(path)
    for name in ["app.py", "requirements.txt", ".env", ".env.example"]:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            assets["config_files"].append(path)
    for name in ["models", "security/secure_backups"]:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            assets["model_paths"].append(path)
    assets["data_flows"] = [
        "User -> Gradio -> generate_text -> DeepSeek/Gemini -> Response",
        "User -> app -> inference/orchestrator -> modelli IA",
        "Security -> guard/verifier -> report.json",
    ]
    return assets


def inventory_dependencies(root: str) -> Dict[str, Any]:
    """Inventario dipendenze da requirements.txt e pip list."""
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "requirements_file": None,
        "requirements": [],
        "pip_list": [],
        "vulnerable_candidates": [],
    }
    req_path = os.path.join(root, "requirements.txt")
    if os.path.isfile(req_path):
        out["requirements_file"] = req_path
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    out["requirements"].append(line)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=root,
        )
        if r.returncode == 0 and r.stdout:
            out["pip_list"] = json.loads(r.stdout)
    except Exception:
        pass
    return out


def run_assessment(root: str = None) -> Dict[str, Any]:
    """Esegue assessment completo e restituisce report."""
    root = root or get_project_root()
    report = {
        "phase": "1_assessment",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "assets": inventory_critical_assets(root),
        "dependencies": inventory_dependencies(root),
    }
    return report


def export_assessment_report(report: Dict[str, Any], path: str = None) -> str:
    """Salva report assessment in JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "assessment_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    r = run_assessment()
    out = export_assessment_report(r)
    print(f"Assessment report: {out}")
