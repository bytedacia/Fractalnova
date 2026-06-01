import os
import json
from watchdog import save_baseline
from guard import run_guard


def main():
    root = os.getenv('SECURITY_ROOT', os.path.dirname(os.path.dirname(__file__)))
    # Se non esiste baseline, creala e termina con successo (prima esecuzione)
    baseline_path = os.path.join(os.path.dirname(__file__), 'baseline.json')
    if not os.path.isfile(baseline_path):
        save_baseline(root)
        print('Baseline created')
        return 0
    # Esegui guard (genera report ed eventualmente cifra file in repo checkout
    report = run_guard(auto_encrypt=False)
    print(f'Report at: {report}')
    # Opzionale: fallire su findings
    try:
        with open(report, 'r', encoding='utf-8') as f:
            data = json.load(f)
        findings = data.get('findings', {})
        changes = data.get('changes', {})
        if (isinstance(changes, dict) and changes and changes.get('status') != 'no_baseline') or findings:
            print('Security findings detected')
            return 1
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


