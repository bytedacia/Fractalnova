# Security Policy

## Versioni supportate
| Versione | Supportata |
|---|---|
| 0.1.x | ✅ |

## Segnalare una vulnerabilità
Scrivi a **security@fractalnova.example** (PGP su richiesta). **Non** aprire issue pubbliche
per vulnerabilità non divulgate.

- **Acknowledgement**: entro 48 ore.
- **Triage + severità (CVSS)**: entro 5 giorni lavorativi.
- **Fix target**: Critical ≤ 7 giorni · High ≤ 30 giorni · Medium ≤ 90 giorni.
- Pratichiamo **coordinated disclosure**: pubblicazione dopo il rilascio della patch.

## Scope
In scope: `api/`, `fractalnova/`, `serving/`, `inference/`, `training/`, infra (Docker/K8s).
Out of scope: attacchi DoS volumetrici, social engineering, dipendenze di terze parti upstream.

## Hardening applicato (vedi `security/AUDIT.md`)
JWT + bcrypt, rate limiting, isolamento per utente, sanitizzazione input, immagini non-root,
segreti via env/secret, nessun prompt "senza limiti", outreach con consenso.
