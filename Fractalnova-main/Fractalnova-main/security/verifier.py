import os
import json
import re
from typing import List


SUSPICIOUS_PATTERNS = [
    r"eval\(",
    r"exec\(",
    r"subprocess\.",
    r"os\.system\(",
    r"__import__\(",
    r"pickle\.loads\(",
    r"base64\.b64decode\(",
]


def scan_paths(paths: List[str]) -> dict:
    findings = {}
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for fn in filenames:
                    _scan_file(os.path.join(dirpath, fn), findings)
        elif os.path.isfile(path):
            _scan_file(path, findings)
    return findings


def _scan_file(fp: str, findings: dict) -> None:
    try:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        for pat in SUSPICIOUS_PATTERNS:
            if re.search(pat, content):
                findings.setdefault(fp, []).append(pat)
    except Exception:
        return


def export_report(findings: dict, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)


