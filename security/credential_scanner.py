"""
Pipeline Fase 1 - Assessment: scansione credenziali hardcoded e secret nel codice.
"""
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple


# Pattern comuni per secret/credenziali (evitare falsi positivi con commenti)
SECRET_PATTERNS = [
    (r"\b(api_key|apikey|api-key)\s*=\s*['\"][^'\"]+['\"]", "API key"),
    (r"\b(secret|password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]", "Password/Secret"),
    (r"\b(token)\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "Token"),
    (r"['\"]sk-[a-zA-Z0-9]{20,}['\"]", "OpenAI-style key"),
    (r"['\"]hf_[a-zA-Z0-9]{20,}['\"]", "HuggingFace token"),
    (r"\b(aws_access_key|aws_secret)\s*=\s*['\"][^'\"]+['\"]", "AWS credential"),
    (r"Bearer\s+[a-zA-Z0-9_.-]+", "Bearer token in code"),
    (r"mongodb(\+srv)?://[^'\"]+:[^@]+@", "MongoDB URI with password"),
    (r"postgres(ql)?://[^'\"]+:[^@]+@", "PostgreSQL URI with password"),
    (r"mysql://[^'\"]+:[^@]+@", "MySQL URI with password"),
    (r"redis://[^'\"]+:[^@]+@", "Redis URI with password"),
]


def scan_file(filepath: str, content: str) -> List[Tuple[int, str, str]]:
    """Restituisce (line_num, pattern_name, snippet) per ogni match."""
    findings = []
    for i, line in enumerate(content.splitlines(), 1):
        # Salta commenti e docstring
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        for pattern, name in SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Non includere il valore nel report
                snippet = line.strip()[:60] + "..." if len(line) > 60 else line.strip()
                findings.append((i, name, snippet))
                break
    return findings


def scan_directory(root: str, extensions: tuple = (".py", ".js", ".ts", ".json", ".env", ".yaml", ".yml")) -> Dict[str, List[dict]]:
    """Scansiona directory per credenziali hardcoded."""
    results = {}
    for dirpath, _, filenames in os.walk(root):
        # Salta venv e .git
        if "venv" in dirpath or ".git" in dirpath or "__pycache__" in dirpath or "node_modules" in dirpath:
            continue
        for name in filenames:
            if not name.endswith(extensions):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            findings = scan_file(path, content)
            if findings:
                rel = os.path.relpath(path, root)
                results[rel] = [{"line": ln, "type": t, "snippet": s} for ln, t, s in findings]
    return results


def run_credential_scan(root: str = None) -> dict:
    """Esegue scan e restituisce report."""
    if root is None:
        root = os.path.dirname(os.path.dirname(__file__))
    findings = scan_directory(root)
    return {
        "phase": "1_credential_scan",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "files_with_findings": len(findings),
        "findings": findings,
    }


if __name__ == "__main__":
    import json
    r = run_credential_scan()
    print(json.dumps(r, indent=2, ensure_ascii=False))
