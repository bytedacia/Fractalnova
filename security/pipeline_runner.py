"""
Esecuzione completa pipeline di sicurezza (tutte le fasi) e report unificato.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List

from .asset_inventory import run_assessment, export_assessment_report
from .credential_scanner import run_credential_scan
from .sast_runner import run_sast
from .dependency_scan import run_dependency_scan
from .security_checklist import checklist_report
from .kpi_report import gather_kpi_data
from .threat_model_template import get_threat_model_template
from .dr_config import get_dr_config
from .restore_runner import list_available_restores


def run_full_pipeline(root: str = None, skip_sast_if_no_bandit: bool = True) -> Dict[str, Any]:
    """
    Esegue tutte le fasi della pipeline e restituisce report unico.
    skip_sast_if_no_bandit: se True non fallisce se Bandit non è installato.
    """
    root = root or os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
    report = {
        "pipeline": "FractalNova Security Pipeline",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root": root,
        "phases": {},
    }
    # Fase 1
    report["phases"]["1_assessment"] = run_assessment(root)
    report["phases"]["1_credential_scan"] = run_credential_scan(root)
    # Fase 5
    report["phases"]["5_sast"] = run_sast(root)
    report["phases"]["5_dependency_scan"] = run_dependency_scan(root)
    # Fase 7
    report["phases"]["7_dr_config"] = get_dr_config()
    report["phases"]["7_restore_list"] = list_available_restores(root)
    # Fase 9-10
    report["phases"]["9_checklist"] = checklist_report()
    report["phases"]["10_kpi"] = gather_kpi_data(root)
    report["phases"]["10_threat_model"] = get_threat_model_template()
    # Summary
    cred_files = report["phases"]["1_credential_scan"].get("files_with_findings", 0)
    sast_high = report["phases"]["5_sast"].get("summary", {}).get("by_severity", {}).get("HIGH", 0)
    vulns = len(report["phases"]["5_dependency_scan"].get("pip_audit", {}).get("vulnerabilities", []))
    report["summary"] = {
        "credential_findings": cred_files,
        "sast_high_findings": sast_high,
        "dependency_vulnerabilities": vulns,
        "checklist_completed": report["phases"]["9_checklist"].get("completed", 0),
        "checklist_total": report["phases"]["9_checklist"].get("total", 0),
    }
    return report


def export_pipeline_report(report: Dict[str, Any] = None, path: str = None) -> str:
    """Salva report pipeline in JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "pipeline_report.json")
    if report is None:
        report = run_full_pipeline()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    r = run_full_pipeline()
    out = export_pipeline_report(r)
    print("Pipeline report:", out)
    print("Summary:", json.dumps(r["summary"], indent=2))
