"""
Pipeline Protezione Codice Malevolo: runner unico per pre-commit e pre-push.
Esegue: secret scan, malware patterns, SAST (Bandit), dependency check, hash check, diff analysis.
Uscita 0 = OK, 1 = blocco (hard block), 2 = warning (soft block).
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Root progetto (parent di security/)
PROJECT_ROOT = os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_staged_files(root: str) -> List[str]:
    """File in staging (path assoluti)."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return []
        return [os.path.normpath(os.path.join(root, f)) for f in (r.stdout or "").strip().splitlines() if f.strip()]
    except Exception:
        return []


def run_secret_scan(root: str, paths: Optional[List[str]] = None) -> Tuple[bool, Dict]:
    """Credential scanner su path o root."""
    try:
        from security.credential_scanner import run_credential_scan, scan_directory
    except ImportError:
        from .credential_scanner import run_credential_scan, scan_directory
    if paths:
        try:
            from security.credential_scanner import scan_file
        except ImportError:
            from .credential_scanner import scan_file
        results = {}
        for p in paths:
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            findings = scan_file(p, content)
            if findings:
                rel = os.path.relpath(p, root)
                results[rel] = [{"line": ln, "type": t, "snippet": s} for ln, t, s in findings]
        ok = len(results) == 0
        return ok, {"files_with_findings": len(results), "findings": results}
    r = run_credential_scan(root)
    return r.get("files_with_findings", 0) == 0, r


def run_malware_scan(paths: Optional[List[str]] = None, root: str = None) -> Tuple[bool, Dict]:
    try:
        from security.malware_patterns import run_malware_scan as _run
    except ImportError:
        from .malware_patterns import run_malware_scan as _run
    root = root or PROJECT_ROOT
    if paths:
        rel_paths = [os.path.relpath(p, root) for p in paths if os.path.isfile(p)]
        report = _run(paths=rel_paths, root=root)
    else:
        report = _run(root=root)
    ok = report.get("files_with_findings", 0) == 0
    return ok, report


def run_sast(root: str) -> Tuple[bool, Dict]:
    try:
        from security.sast_runner import run_sast as _run
    except ImportError:
        from .sast_runner import run_sast as _run
    report = _run(root)
    high = report.get("summary", {}).get("by_severity", {}).get("HIGH", 0)
    return high == 0, report


def run_dependency_check(root: str) -> Tuple[bool, Dict]:
    try:
        from security.dependency_scan import run_dependency_scan
    except ImportError:
        from .dependency_scan import run_dependency_scan
    report = run_dependency_scan(root)
    vulns = report.get("pip_audit", {}).get("vulnerabilities", [])
    return len(vulns) == 0, report


def run_hash_check(root: str, paths: Optional[List[str]] = None) -> Tuple[bool, Dict]:
    try:
        from security.hash_checker import run_hash_check as _run
    except ImportError:
        from .hash_checker import run_hash_check as _run
    rel_paths = [os.path.relpath(p, root) for p in paths] if paths else None
    report = _run(root, paths=rel_paths)
    result = report.get("result", {})
    ok = result.get("ok", True)
    return ok, report


def run_diff_analysis(root: str) -> Tuple[bool, Dict]:
    try:
        from security.diff_analyzer import run_diff_analysis
    except ImportError:
        from .diff_analyzer import run_diff_analysis
    report = run_diff_analysis(root)
    return not report.get("block", False), report


def run_pre_commit(
    root: str = None,
    skip_sast: bool = False,
    skip_deps: bool = False,
    soft_block: bool = False,
    staged_only: bool = True,
) -> Dict:
    """
    Esegue tutti i check pre-commit.
    staged_only: se True, secret/malware scan solo su file staged.
    soft_block: se True, fallimenti danno exit 2 (warning) invece di 1 (block).
    """
    root = root or PROJECT_ROOT
    staged = get_staged_files(root) if staged_only else None
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "staged_count": len(staged) if staged else None,
        "checks": {},
        "block_reasons": [],
        "warnings": [],
    }
    all_ok = True

    # 1. Secret scanning
    ok, data = run_secret_scan(root, staged)
    report["checks"]["secret_scan"] = {"ok": ok, "data": data}
    if not ok:
        report["block_reasons"].append("Secret/credenziali rilevati nel codice")
        all_ok = False

    # 2. Malware/backdoor patterns
    ok, data = run_malware_scan(staged, root)
    report["checks"]["malware_patterns"] = {"ok": ok, "data": data}
    if not ok:
        report["block_reasons"].append("Pattern malevoli/backdoor rilevati")
        all_ok = False

    # 3. SAST (Bandit)
    if not skip_sast:
        ok, data = run_sast(root)
        report["checks"]["sast"] = {"ok": ok, "data": data}
        if not ok:
            report["block_reasons"].append("SAST: vulnerabilità HIGH")
            all_ok = False
    else:
        report["checks"]["sast"] = {"ok": True, "data": {"skipped": True}}

    # 4. Dependency check
    if not skip_deps:
        ok, data = run_dependency_check(root)
        report["checks"]["dependency"] = {"ok": ok, "data": data}
        if not ok:
            report["warnings"].append("Dipendenze con vulnerabilità note (pip audit)")
            if not soft_block:
                all_ok = False
    else:
        report["checks"]["dependency"] = {"ok": True, "data": {"skipped": True}}

    # 5. Hash check (solo se esiste baseline)
    ok, data = run_hash_check(root, staged)
    report["checks"]["hash_check"] = {"ok": ok, "data": data}
    if not ok and data.get("result", {}).get("changed"):
        report["block_reasons"].append("Hash file modificati rispetto a baseline")
        all_ok = False

    # 6. Diff analysis
    ok, data = run_diff_analysis(root)
    report["checks"]["diff_analysis"] = {"ok": ok, "data": data}
    if not ok:
        report["block_reasons"].append("Modifiche sospette nel diff (auth/endpoint/eval)")
        all_ok = False

    report["all_ok"] = all_ok
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(description="FractalNova pre-commit security runner")
    p.add_argument("--root", default=PROJECT_ROOT, help="Project root")
    p.add_argument("--skip-sast", action="store_true", help="Skip Bandit SAST")
    p.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    p.add_argument("--soft", action="store_true", help="Soft block: warnings only, exit 2")
    p.add_argument("--json", action="store_true", help="Output report JSON only")
    p.add_argument("--pre-push", action="store_true", help="Pre-push: scan full repo (not only staged)")
    args = p.parse_args()
    report = run_pre_commit(
        root=args.root,
        skip_sast=args.skip_sast,
        skip_deps=args.skip_deps,
        soft_block=args.soft,
        staged_only=not args.pre_push,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("FractalNova - Pre-commit security scan")
        print("Root:", report["root"])
        for name, c in report["checks"].items():
            status = "OK" if c.get("ok") else "FAIL"
            print(f"  {name}: {status}")
        if report.get("block_reasons"):
            print("Block reasons:", report["block_reasons"])
        if report.get("warnings"):
            print("Warnings:", report["warnings"])
        print("Result:", "PASS" if report["all_ok"] else "BLOCK")
    if report["all_ok"]:
        sys.exit(0)
    sys.exit(2 if args.soft else 1)


if __name__ == "__main__":
    main()
