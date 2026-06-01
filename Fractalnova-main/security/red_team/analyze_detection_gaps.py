"""
Analizza red_team_report.json: gap di detection, priorità, metriche.
"""
import os
import json
import sys
from typing import Dict, List


def load_report(path: str = None) -> Dict:
    red_team_dir = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(red_team_dir, "red_team_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(report: Dict) -> Dict:
    """Gap analysis e metriche."""
    if not report or "details" not in report:
        return {"error": "Report vuoto o mancante"}
    details = report["details"]
    bypassed = [d for d in details if d.get("blocked") is False]
    blocked = [d for d in details if d.get("blocked") is True]
    critical_bypassed = [d for d in bypassed if d.get("critical")]

    by_stage = {}
    for d in blocked:
        for s in d.get("stages") or [d.get("stage") or "unknown"]:
            by_stage[s] = by_stage.get(s, 0) + 1

    return {
        "summary": {
            "total": report.get("total_tests", 0),
            "blocked": report.get("blocked", 0),
            "bypassed": report.get("bypassed", 0),
            "detection_rate_pct": report.get("detection_rate_pct", 0),
        },
        "critical_gaps": [
            {"payload_id": d["payload_id"], "type": d["type"]}
            for d in critical_bypassed
        ],
        "bypassed_payloads": [
            {"payload_id": d["payload_id"], "type": d["type"], "critical": d.get("critical")}
            for d in bypassed
        ],
        "detection_by_layer": by_stage,
        "recommendations": _recommendations(bypassed, critical_bypassed),
    }


def _recommendations(bypassed: List[Dict], critical_bypassed: List[Dict]) -> List[str]:
    rec = []
    if critical_bypassed:
        types = set(d["type"] for d in critical_bypassed)
        if any("Secret" in t for t in types):
            rec.append("Rafforzare credential_scanner: pattern aggiuntivi per secret in chiaro.")
        if any("Backdoor" in t or "Shell" in t for t in types):
            rec.append("Aggiungere pattern subprocess/exec/shell in malware_patterns e SAST.")
        if any("Exfiltr" in t for t in types):
            rec.append("Pattern requests.post/upload verso URL esterni in malware_patterns.")
        if any("Obfusc" in t or "eval" in t for t in types):
            rec.append("Pattern base64.decode + exec/eval e eval/exec in malware_patterns.")
        rec.append("Priorità massima: payload critici non bloccati vanno coperti da regole.")
    if bypassed and not critical_bypassed:
        rec.append("Tuning: ridurre falsi negativi per payload non critici.")
    if not bypassed:
        rec.append("Nessun gap: detection efficace su tutti i payload testati.")
    return rec


def main():
    report = load_report()
    if not report:
        print("Nessun report trovato. Esegui prima: python -m security.red_team.red_team_orchestrator")
        sys.exit(1)
    a = analyze(report)
    print(json.dumps(a, indent=2, ensure_ascii=False))
    if a.get("critical_gaps"):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
