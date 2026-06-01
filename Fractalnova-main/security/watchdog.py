import os
import hashlib
import json
from typing import Dict


BASELINE_FILE = os.getenv("SECURITY_BASELINE", os.path.join(os.path.dirname(__file__), "baseline.json"))


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_baseline(root: str) -> Dict[str, str]:
    records: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        if ".git" in dirpath or "models" in dirpath:
            continue
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                records[os.path.relpath(fp, root)] = _hash_file(fp)
            except Exception:
                continue
    return records


def save_baseline(root: str) -> None:
    records = build_baseline(root)
    with open(BASELINE_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def detect_changes(root: str) -> Dict[str, str]:
    try:
        with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
            base = json.load(f)
    except FileNotFoundError:
        return {"status": "no_baseline"}

    current = build_baseline(root)
    changed = {}
    for path, h in current.items():
        if path not in base:
            changed[path] = "new"
        elif base[path] != h:
            changed[path] = "modified"
    for path in base:
        if path not in current:
            changed[path] = "deleted"
    return changed


