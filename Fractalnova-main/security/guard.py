import os
from typing import Optional
from .encryptor import encrypt_file
from .watchdog import detect_changes
from .verifier import scan_paths, export_report


SECURITY_ROOT = os.getenv("SECURITY_ROOT", os.path.dirname(os.path.dirname(__file__)))
SECURITY_REPORT = os.getenv("SECURITY_REPORT", os.path.join(os.path.dirname(__file__), "report.json"))


def run_guard(auto_encrypt: bool = True) -> Optional[str]:
    """
    Esegue verifiche di integrità e pattern sospetti. Se trova modifiche o pattern,
    può cifrare i file bersaglio come misura estrema (opt-in).
    """
    changes = detect_changes(SECURITY_ROOT)
    findings = scan_paths([SECURITY_ROOT])
    export_report({"changes": changes, "findings": findings}, SECURITY_REPORT)

    if auto_encrypt and (changes and changes != {"status": "no_baseline"} or findings):
        # Cifra file sensibili
        targets = [
            os.path.join(SECURITY_ROOT, "inference", "generate.py"),
            os.path.join(SECURITY_ROOT, "inference", "orchestrator.py"),
        ]
        for t in targets:
            try:
                encrypt_file(t)
            except Exception:
                continue
        return SECURITY_REPORT
    return SECURITY_REPORT


