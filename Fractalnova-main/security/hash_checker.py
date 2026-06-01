"""
Pipeline Protezione Codice Malevolo: verifica hash (SHA256) file.
Baseline locale; opzionale verifica contro VirusTotal/Abuse.ch (API key richiesta).
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_hashes.json")


def sha256_file(path: str) -> Optional[str]:
    """Calcola SHA256 di un file."""
    if not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def build_baseline(root: str, extensions: tuple = (".py", ".json", ".yml", ".yaml", ".sh", ".ps1", ".md")) -> Dict:
    """Costruisce baseline hash per tutti i file rilevanti sotto root."""
    baseline = {"version": 1, "root": root, "timestamp": datetime.utcnow().isoformat() + "Z", "hashes": {}}
    for dirpath, _, filenames in os.walk(root):
        if "venv" in dirpath or ".git" in dirpath or "__pycache__" in dirpath or "node_modules" in dirpath:
            continue
        for name in filenames:
            if not name.endswith(extensions):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            digest = sha256_file(path)
            if digest:
                baseline["hashes"][rel] = digest
    return baseline


def load_baseline(path: str = None) -> Optional[Dict]:
    """Carica baseline da file."""
    path = path or BASELINE_PATH
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_hashes(root: str, baseline: Dict = None, paths: List[str] = None) -> Dict:
    """
    Confronta hash attuali con baseline.
    paths: se fornito, verifica solo questi file (relativi a root).
    Restituisce { "ok": bool, "missing": [], "changed": [], "new": [], "details": {} }
    """
    baseline = baseline or load_baseline()
    if not baseline:
        return {"ok": True, "missing": [], "changed": [], "new": [], "details": {}, "message": "No baseline"}
    expected = baseline.get("hashes", {})
    missing = []
    changed = []
    new = []
    to_check = list(expected.keys()) if not paths else [p for p in paths if p in expected]
    if paths:
        for p in paths:
            if p not in expected and os.path.isfile(os.path.join(root, p)):
                new.append(p)
    for rel in to_check:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        current = sha256_file(full)
        if current != expected.get(rel):
            changed.append(rel)
    for rel in expected:
        if rel not in to_check and not paths:
            full = os.path.join(root, rel)
            if os.path.isfile(full):
                current = sha256_file(full)
                if current != expected.get(rel):
                    changed.append(rel)
    ok = len(missing) == 0 and len(changed) == 0
    return {
        "ok": ok,
        "missing": missing,
        "changed": changed,
        "new": new,
        "details": {"missing": len(missing), "changed": len(changed), "new": len(new)},
    }


def run_hash_check(root: str = None, baseline_path: str = None, paths: List[str] = None) -> Dict:
    """Report per pipeline pre-commit."""
    root = root or os.getcwd()
    baseline = load_baseline(baseline_path) if baseline_path else load_baseline()
    result = check_hashes(root, baseline, paths)
    return {
        "phase": "hash_check",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "baseline_loaded": baseline is not None,
        "result": result,
    }


def save_baseline(baseline: Dict, path: str = None) -> str:
    path = path or BASELINE_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    return path


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    if "--build" in sys.argv:
        b = build_baseline(root)
        save_baseline(b)
        print("Baseline salvata:", BASELINE_PATH)
    else:
        r = run_hash_check(root)
        print(json.dumps(r, indent=2))
        exit(0 if r.get("result", {}).get("ok", True) else 1)
