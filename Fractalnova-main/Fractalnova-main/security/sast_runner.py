"""
Pipeline Fase 5 - Sicurezza codice: esecuzione SAST (Bandit) e report.
"""
import os
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_project_root() -> str:
    return os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))


def run_bandit(path: str = None, config: str = None) -> Dict[str, Any]:
    """
    Esegue Bandit su path. Restituisce report dict.
    Se Bandit non è installato: pip install bandit
    """
    path = path or get_project_root()
    cmd = [sys.executable, "-m", "bandit", "-r", path, "-f", "json", "-q"]
    if config and os.path.isfile(config):
        cmd.extend(["-c", config])
    # Escludi venv e .git
    exclude = os.path.join(path, "venv"), os.path.join(path, ".git")
    for e in exclude:
        if os.path.exists(e):
            cmd.extend(["--exclude", e])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=path)
        if r.returncode in (0, 1):  # 1 = findings
            try:
                return json.loads(r.stdout or "{}")
            except json.JSONDecodeError:
                return {"results": [], "errors": [r.stderr or "Bandit output not JSON"]}
        return {"results": [], "errors": [r.stderr or f"Exit code {r.returncode}"]}
    except subprocess.TimeoutExpired:
        return {"results": [], "errors": ["Bandit timeout"]}
    except FileNotFoundError:
        return {"results": [], "errors": ["Bandit non installato: pip install bandit"]}
    except Exception as e:
        return {"results": [], "errors": [str(e)]}


def run_sast(root: str = None) -> Dict[str, Any]:
    """Report SAST per pipeline (Bandit)."""
    root = root or get_project_root()
    bandit = run_bandit(root)
    return {
        "phase": "5_sast",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": "bandit",
        "path": root,
        "bandit": bandit,
        "summary": {
            "total_findings": len(bandit.get("results", [])),
            "by_severity": _count_by_severity(bandit.get("results", [])),
        },
    }


def _count_by_severity(results: List[Dict]) -> Dict[str, int]:
    c = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results:
        sev = r.get("issue_severity", "LOW")
        c[sev] = c.get(sev, 0) + 1
    return c


if __name__ == "__main__":
    import json as j
    r = run_sast()
    print(j.dumps(r, indent=2, ensure_ascii=False))
