# Red Team Testing - FractalNova

**⚠️ SOLO in ambiente di test isolato (VM/container, branch dedicato). Non eseguire in produzione.**

Payload e script servono a **validare la detection** (pre-commit, SAST, secret scan).  
I file in `payloads/` contengono codice malevolo **intenzionale** per test; non eseguirli come parte dell'applicazione.

## Uso

```bash
# Dalla root del repo (ambiente di test)
python -m security.red_team.red_team_orchestrator

# Analisi gap
python -m security.red_team.analyze_detection_gaps
```

## Struttura

- `payloads/` – snippet malevoli per categoria (secret, backdoor, reverse_shell, …)
- `red_team_orchestrator.py` – inietta payload in target di test, esegue detection, genera report
- `analyze_detection_gaps.py` – analizza `red_team_report.json`
- `config.json` – elenco test (payload, target, messaggio)
