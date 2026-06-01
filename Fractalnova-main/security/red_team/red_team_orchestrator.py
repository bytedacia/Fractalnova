"""
Pipeline Red Team - Simulazione attacco con payload controllati.
Testa la detection (secret scan, malware patterns) senza eseguire git commit.
Solo per ambiente di test isolato.
"""
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Root repo (parent di security/red_team/)
REPO_ROOT = os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RED_TEAM_DIR = os.path.dirname(os.path.abspath(__file__))
PAYLOADS_DIR = os.path.join(RED_TEAM_DIR, "payloads")


def _load_config() -> Dict:
    path = os.path.join(RED_TEAM_DIR, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _test_target_path() -> str:
    return os.path.join(REPO_ROOT, "security", "red_team", "_test_target.py")


def _run_secret_scan_on_file(root: str, filepath: str) -> Tuple[bool, Dict]:
    """Esegue credential scanner sul singolo file."""
    sys.path.insert(0, root)
    try:
        from security.credential_scanner import scan_file
    except ImportError:
        from security.credential_scanner import scan_file
    if not os.path.isfile(filepath):
        return True, {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    findings = scan_file(filepath, content)
    ok = len(findings) == 0
    return ok, {"findings": [{"line": ln, "type": t, "snippet": s[:80]} for ln, t, s in findings]}


def _run_malware_scan_on_file(root: str, filepath: str) -> Tuple[bool, Dict]:
    """Esegue malware pattern scan sul singolo file."""
    sys.path.insert(0, root)
    try:
        from security.malware_patterns import scan_file
    except ImportError:
        from security.malware_patterns import scan_file
    findings = scan_file(filepath)
    ok = len(findings) == 0
    return ok, {"findings": [{"pattern": n, "line": ln, "snippet": s[:80]} for n, s, ln in findings]}


def run_single_test(
    payload_id: str,
    payload_type: str,
    payload_content: str,
    target_path: str,
    root: str,
) -> Dict:
    """
    Scrive payload nel target, esegue detection, restituisce risultato.
    Non fa commit; solo scrittura file + run scanner + restore.
    """
    original = ""
    if os.path.isfile(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            original = f.read()
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(payload_content)
        secret_ok, secret_data = _run_secret_scan_on_file(root, target_path)
        malware_ok, malware_data = _run_malware_scan_on_file(root, target_path)
        blocked = not (secret_ok and malware_ok)
        stage = []
        if not secret_ok:
            stage.append("secret_scan")
        if not malware_ok:
            stage.append("malware_patterns")
        return {
            "payload_id": payload_id,
            "type": payload_type,
            "blocked": blocked,
            "stage": stage[0] if stage else None,
            "stages": stage,
            "secret_scan_ok": secret_ok,
            "malware_scan_ok": malware_ok,
            "secret_findings": secret_data.get("findings", []),
            "malware_findings": malware_data.get("findings", []),
            "time_detection_sec": 0,
        }
    finally:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(original)


def run_test_suite(root: str = None) -> Dict:
    """Esegue suite completa: un test per ogni payload in config."""
    root = os.path.abspath(root or REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    config = _load_config()
    target_path = _test_target_path()
    results = []
    for p in config["payloads"]:
        payload_file = os.path.join(PAYLOADS_DIR, p["file"])
        if not os.path.isfile(payload_file):
            results.append({
                "payload_id": p["id"],
                "type": p["type"],
                "blocked": None,
                "stage": "skip",
                "error": "payload file not found",
            })
            continue
        with open(payload_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        t0 = datetime.utcnow()
        r = run_single_test(
            p["id"],
            p["type"],
            content,
            target_path,
            root,
        )
        t1 = datetime.utcnow()
        r["time_detection_sec"] = (t1 - t0).total_seconds()
        r["critical"] = p.get("critical", False)
        results.append(r)
        print(f"  {p['id']} {p['type']}: {'BLOCKED' if r['blocked'] else 'BYPASSED'} ({r.get('stage') or '-'})")

    blocked_count = sum(1 for x in results if x.get("blocked") is True)
    bypassed_count = sum(1 for x in results if x.get("blocked") is False)
    report = {
        "test_date": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "total_tests": len(results),
        "blocked": blocked_count,
        "bypassed": bypassed_count,
        "skipped": len(results) - blocked_count - bypassed_count,
        "detection_rate_pct": round(100.0 * blocked_count / len(results), 1) if results else 0,
        "details": results,
    }
    return report


def save_report(report: Dict, path: str = None) -> str:
    path = path or os.path.join(RED_TEAM_DIR, "red_team_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fractal Nova Red Team - test detection pipeline")
    p.add_argument("--root", default=REPO_ROOT, help="Repo root")
    p.add_argument("--output", default=None, help="Report output path")
    args = p.parse_args()
    print("Red Team - Running detection tests (no commit)...")
    report = run_test_suite(args.root)
    out = save_report(report, args.output)
    print(f"Blocked: {report['blocked']}/{report['total_tests']} | Detection rate: {report['detection_rate_pct']}%")
    print(f"Report: {out}")
    return 0 if report["bypassed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
