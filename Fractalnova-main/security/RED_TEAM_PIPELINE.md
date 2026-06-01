# Pipeline Red Team Testing - FractalNova

Simulazione attacco con codice malevolo controllato per **validare la detection** (pre-commit, secret scan, malware patterns).  
**Solo in ambiente di test isolato (VM/container, branch `security-testing`).**

## 1. Preparazione ambiente

- Clone/branch dedicato, es. `security-testing`
- Network isolata; snapshot per rollback
- Alert team che è in corso test autorizzato

## 2. Payload (controllati)

In `security/red_team/payloads/`:

| ID   | Tipo              | File                    | Critico |
|------|-------------------|-------------------------|---------|
| P001 | Secret Leakage    | payload_secrets.py      | sì      |
| P002 | Backdoor          | payload_backdoor.py     | sì      |
| P003 | Reverse Shell     | payload_reverse_shell.py| sì      |
| P004 | Data Exfiltration | payload_exfiltration.py | sì      |
| P005 | Cryptominer       | payload_miner.py        | sì      |
| P006 | Offuscamento      | payload_obfuscated.py   | sì      |
| P007 | eval/exec         | payload_eval_exec.py     | sì      |
| P008 | Sensitive Path I/O| payload_sensitive_path.py| sì      |

## 3. Esecuzione test

```bash
# Dalla root del repo (ambiente di test)
python -m security.red_team.red_team_orchestrator
```

- Scrive ogni payload in `security/red_team/_test_target.py`
- Esegue **secret scan** e **malware patterns** su quel file
- Registra: bloccato / bypassed, layer che ha rilevato (secret_scan, malware_patterns)
- Ripristina `_test_target.py` dopo ogni test
- Genera `security/red_team/red_team_report.json`

## 4. Analisi gap

```bash
python -m security.red_team.analyze_detection_gaps
```

- Legge `red_team_report.json`
- Elenca payload bypassati, gap critici, detection per layer
- Suggerisce azioni (rafforzare credential_scanner, malware_patterns, SAST)

## 5. Metriche

- **Detection rate**: % payload bloccati (target > 95%)
- **Time to detect**: tempo medio (obiettivo < 30 s)
- **Critical bypassed**: zero per payload critici

## 6. CI (opzionale)

Workflow `.github/workflows/red-team.yml`: esecuzione schedulata (es. lunedì 2:00) + manuale; upload report e notifica in caso di gap.

## 7. Fasi avanzate (manuali)

- **Evasion**: offuscare payload e ritestare
- **Social engineering**: PR con titolo “bugfix” e payload nascosto
- **Persistence**: verificare che la detection blocchi anche dopo merge/rebase

Vedi anche `security/red_team/README.md`.
