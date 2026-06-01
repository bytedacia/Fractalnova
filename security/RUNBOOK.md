# Runbook — Sicurezza FractalNova

Operazioni ricorrenti e risposta agli incidenti. Per il piano completo vedi [SECURITY_PIPELINE.md](SECURITY_PIPELINE.md).

---

## Operazioni giornaliere / settimanali

| Azione | Comando / modulo | Frequenza |
|--------|-------------------|-----------|
| Avvio guard continuo | `python -m security.start_ultra_security` | Quando il sistema è in uso |
| Verifica integrità + report | `python -c "from security.guard import run_guard; run_guard(auto_encrypt=False)"` | Giornaliera |
| Controllo backup | `SecureBackupManager` (automatico se ultra_guard attivo) | Verifica settimanale che i backup siano presenti |
| Dependency check | `pip audit` o Snyk/Dependabot | Settimanale / in CI |

---

## Risposta a incidente sospetto

1. **Non modificare** file o log prima di averne copia.
2. **Esegui verifica:**  
   `python -c "from security.guard import run_guard; run_guard(auto_encrypt=False)"`  
   Controlla `security/report.json`.
3. **Verifica avanzata:** avvia `start_ultra_security` e controlla log in `security/` (es. `emergency_log.json` se presente).
4. **Isola:** se in produzione, disconnetti il servizio dalla rete o metti in modalità manutenzione.
5. **Documenta:** ora, azioni eseguite, file/endpoint coinvolti.
6. **Ripristino:** se necessario, usa backup da `security/secure_backups` (vedi `backup_manager.py`).
7. **Escalation:** contattare i referenti di emergenza definiti nella Sezione 9 della pipeline.

---

## Contatti di emergenza (da compilare)

- **Security lead:** _________________
- **DevOps / Infra:** _________________
- **Data protection / DPO:** _________________

---

## RTO / RPO (da allineare al piano DR)

- **RTO (Recovery Time Objective):** ______ ore
- **RPO (Recovery Point Objective):** ______ ore

Aggiornare dopo i test di restore (Sezione 7 della pipeline).
