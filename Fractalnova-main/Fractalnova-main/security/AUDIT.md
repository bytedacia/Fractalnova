# FractalNova · Security Audit (v0.1)

> Audit onesto: distingue ciò che è **verificato nel codice** da ciò che **richiede
> verifica in esecuzione**. Niente checkmark non meritati.

## 1. Correzioni applicate (verificabili nel diff)
| Area | Prima | Ora |
|---|---|---|
| Prompt "senza censura/contenuti estremi" | presente in `generate.py` | ✅ rimosso, funzione resa neutra |
| Iniezione di refusi nel testo | `add_typos()` corrompeva l'output | ✅ no-op |
| Outreach email editori | invio **automatico** (spam/rischio legale) | ✅ **opt-in** (`ENABLE_AUTO_OUTREACH`) + piano con consenso |
| Regex estrazione email | rotta (`\\.`) | ✅ corretta |
| Auth API | assente | ✅ JWT (HS256) + bcrypt + OAuth2 |
| Input utente API | non validato | ✅ sanitizzazione + limiti (pydantic + validator) |
| Isolamento dati | n/d | ✅ job/libri filtrati per `user_id` |
| Container | root, Python 3.9 | ✅ non-root, 3.11, healthcheck |
| Segreti | rischio hardcode | ✅ solo `.env`/K8s Secret (in `.gitignore`) |
| Download file | n/d | ✅ servito solo al proprietario, path da whitelist DB |

## 2. Difese a runtime (nel percorso di richiesta)
- **API** (`api/`): JWT, rate limit per utente, CORS configurabile, validazione pydantic.
- **Flask legacy** (`inference/generate.py`): security headers, rate limit per IP, CORS allowlist,
  `MAX_CONTENT_LENGTH`, API key opzionale, `trust_remote_code` off di default.

## 3. Suite `security/` — inventario e stato
La cartella `security/` contiene ~50 moduli. **Constatazione**: sono in larga parte **script
standalone** (scansione, monitoraggio, backup, DR) **non cablati** nel percorso di richiesta
dell'app. Utili come tooling CI/ops, ma non vanno confusi con difese runtime.

| Categoria | File (esempi) | Stato |
|---|---|---|
| SAST / scansione codice | `advanced_verifier.py`, `sast_runner.py`, `malware_patterns.py` | tooling, ⚠️ da validare in CI |
| Integrità / self-monitor | `self_monitor.py`, `watchdog.py`, `hash_checker.py` | tooling ops, ⚠️ non runtime |
| Orchestrazione | `ultra_guard.py`, `start_ultra_security.py`, `pipeline_runner.py` | script, ⚠️ da verificare |
| Crittografia / backup / DR | `encryptor.py`, `backup_manager.py`, `dr_config.py`, `restore_runner.py` | tooling, ⚠️ test richiesti |
| App-security riusabili | `input_sanitizer.py`, `prompt_guard.py`, `output_sanitizer.py`, `security_headers.py`, `model_rate_limiter.py` | ✅ usati da `app.py` (Gradio) |
| Red team | `red_team/` + payload | tooling offensivo, ⚠️ isolare/CI dedicata |

## 4. Raccomandazioni prioritarie
1. **P1** — Cablare `input_sanitizer`/`prompt_guard`/`output_sanitizer` anche nel percorso API
   (oggi solo in `app.py`).
2. **P1** — Ruotare/forzare `API_JWT_SECRET` forte in prod (no default).
3. **P2** — Rate limit distribuito (Redis) al posto di quello in-memory per multi-replica.
4. **P2** — Far girare SAST (`sast_runner`) in CI e fallire la build su finding critici.
5. **P3** — Spostare gli export su storage oggetti (S3) con URL firmati a scadenza.
6. **P3** — Dependency scanning (`dependency_scan.py`) schedulato + Dependabot.

## 5. Da fare per certificazione (SOC2/ISO)
Audit log centralizzato, gestione segreti (Vault/KMS), MFA, penetration test esterno,
data retention/GDPR (`gdpr_utils.py` da integrare nelle API).
