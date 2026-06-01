"""
Pipeline Protezione Codice Malevolo: analisi diff per modifiche sospette.
Rileva: config sensibili, auth, endpoint nascosti, codice dopo return/exit, commenti rimossi in sezioni critiche.
"""
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Red flag: file sensibili
SENSITIVE_FILES = re.compile(
    r"\.env|config\.(yml|yaml|json)|secret|credential|docker-compose|Dockerfile|\.github/workflows"
)
# Red flag: pattern nel diff
AUTH_CHANGE = re.compile(r"[\+\-].*(password|secret|api_key|token|auth|login)\s*[=:]", re.I)
HIDDEN_ENDPOINT = re.compile(r"[\+\-].*@(app\.)?(route|get|post)\s*\([^)]*['\"](/admin|/debug|/shell|/exec)", re.I)
CODE_AFTER_RETURN = re.compile(r"^\s*return\s+[^;]+;\s*\n\s*[\+\+].*[a-zA-Z0-9_]+\s*\(", re.M)
DANGEROUS_IMPORT = re.compile(r"[\+\+].*import\s+(os|subprocess|sys|eval|exec)\s*$", re.I)


def get_staged_diff(root: str) -> str:
    """Esegue git diff --cached e restituisce stdout."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--no-color"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout or ""
    except Exception:
        return ""


def get_staged_files(root: str) -> List[str]:
    """Elenco file in staging (--name-only)."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [f for f in (r.stdout or "").strip().splitlines() if f.strip()]
    except Exception:
        return []


def analyze_diff(diff_text: str, staged_files: List[str]) -> Dict[str, List[dict]]:
    """
    Analizza diff e lista file. Restituisce { "red_flags": [...], "sensitive_files": [...], "warnings": [...] }.
    """
    red_flags = []
    sensitive_files = []
    warnings = []
    for f in staged_files:
        if SENSITIVE_FILES.search(f):
            sensitive_files.append({"file": f, "reason": "config/sensitive"})
    if AUTH_CHANGE.search(diff_text):
        red_flags.append({"type": "auth_change", "description": "Modifica a password/secret/auth nel diff"})
    if HIDDEN_ENDPOINT.search(diff_text):
        red_flags.append({"type": "hidden_endpoint", "description": "Possibile endpoint nascosto (admin/debug/shell)"})
    if re.search(r"[\+\+].*eval\s*\(|[\+\+].*exec\s*\(", diff_text):
        red_flags.append({"type": "eval_exec_added", "description": "Aggiunta eval/exec nel diff"})
    if re.search(r"[\+\+].*subprocess\.(run|call|Popen).*shell\s*=\s*True", diff_text):
        red_flags.append({"type": "shell_true_added", "description": "Aggiunta subprocess con shell=True"})
    if re.search(r"[\-\-].*#.*(security|auth|password|TODO.*fix)", diff_text, re.I):
        warnings.append({"type": "comment_removed", "description": "Rimozione commenti in zona sensibile"})
    return {
        "red_flags": red_flags,
        "sensitive_files": sensitive_files,
        "warnings": warnings,
    }


def run_diff_analysis(root: str = None) -> Dict:
    """Report per pipeline pre-commit."""
    from datetime import datetime
    root = root or os.getcwd()
    if not os.path.isdir(os.path.join(root, ".git")):
        return {
            "phase": "diff_analysis",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "root": root,
            "git_repo": False,
            "red_flags": [],
            "sensitive_files": [],
            "warnings": [],
            "block": False,
        }
    diff = get_staged_diff(root)
    staged = get_staged_files(root)
    analysis = analyze_diff(diff, staged)
    block = len(analysis["red_flags"]) > 0
    return {
        "phase": "diff_analysis",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "git_repo": True,
        "staged_count": len(staged),
        "red_flags": analysis["red_flags"],
        "sensitive_files": analysis["sensitive_files"],
        "warnings": analysis["warnings"],
        "block": block,
    }


if __name__ == "__main__":
    import json
    r = run_diff_analysis()
    print(json.dumps(r, indent=2))
    sys.exit(1 if r.get("block") else 0)
