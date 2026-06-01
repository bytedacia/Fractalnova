"""
Pipeline Fase 5 - Sicurezza codice: dependency scanning (pip audit / requirements check).
"""
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List


def get_project_root() -> str:
    return os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))


def run_pip_audit(root: str = None) -> Dict[str, Any]:
    """
    Esegue 'pip audit' se disponibile (Python 3.11+ o pip-audit installato).
    Restituisce report vulnerabilità dipendenze.
    """
    root = root or get_project_root()
    out = {"vulnerabilities": [], "errors": [], "audit_available": False}
    # pip audit (built-in da 3.11) o pip-audit
    for cmd in [
        [sys.executable, "-m", "pip", "audit", "--format", "json"],
        [sys.executable, "-m", "pip_audit", "--format", "json"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=root)
            out["audit_available"] = True
            if r.returncode != 0 and r.stdout:
                import json
                data = json.loads(r.stdout)
                out["vulnerabilities"] = data.get("vulnerabilities", [])
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue
    return out


def run_dependency_scan(root: str = None) -> Dict[str, Any]:
    """Report dependency scan per pipeline."""
    root = root or get_project_root()
    audit = run_pip_audit(root)
    req_path = os.path.join(root, "requirements.txt")
    requirements = []
    if os.path.isfile(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return {
        "phase": "5_dependency_scan",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": root,
        "requirements_count": len(requirements),
        "pip_audit": audit,
        "summary": {
            "vulnerability_count": len(audit.get("vulnerabilities", [])),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_dependency_scan(), indent=2, ensure_ascii=False))
