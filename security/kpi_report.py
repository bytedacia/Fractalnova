"""
Pipeline Fase 10 - Continuous improvement: metriche KPI sicurezza.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List

from .asset_inventory import run_assessment
from .credential_scanner import run_credential_scan
from .sast_runner import run_sast
from .dependency_scan import run_dependency_scan
from .security_checklist import checklist_report


def gather_kpi_data(root: str = None) -> Dict[str, Any]:
    """Raccoglie dati da assessment, SAST, dependency, checklist."""
    root = root or os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
    assessment = run_assessment(root)
    cred_scan = run_credential_scan(root)
    sast = run_sast(root)
    deps = run_dependency_scan(root)
    checklist = checklist_report()
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "kpi": {
            "credential_findings": cred_scan.get("files_with_findings", 0),
            "sast_high": sast.get("summary", {}).get("by_severity", {}).get("HIGH", 0),
            "sast_medium": sast.get("summary", {}).get("by_severity", {}).get("MEDIUM", 0),
            "dependency_vulnerabilities": deps.get("pip_audit", {}).get("vulnerabilities", []).__len__(),
            "checklist_completed": checklist.get("completed", 0),
            "checklist_total": checklist.get("total", 0),
        },
        "raw": {
            "assessment": assessment,
            "credential_scan": cred_scan,
            "sast": sast,
            "dependency_scan": deps,
            "checklist": checklist,
        },
    }


def export_kpi_report(path: str = None) -> str:
    """Genera e salva report KPI in JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "kpi_report.json")
    data = gather_kpi_data()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    print(export_kpi_report())
