# Pipeline di Protezione da Codice Malevolo — Pre-Commit e Pre-Push

## 1. PRE-COMMIT HOOKS LOCALI

**Hook Git (FractalNova):**
- **pre-commit**: analisi statica prima del commit (secret, malware, SAST, hash, diff)
- **pre-push**: scansione completa repo prima del push

**Funzioni chiave (implementate in `security/`):**
- **Secret scanning** (`credential_scanner.py`, `pre_commit_runner.py`)
  - Ricerca API keys, password, token, chiavi private
  - Blocco commit se trovati segreti

- **Malware/backdoor detection** (`malware_patterns.py`)
  - Pattern webshell, reverse shell, eval/exec, offuscamento (base64, chr, getattr)
  - Rilevamento I/O su percorsi sensibili (/etc, .ssh)
  - Indicatori cryptomining/bot

- **Dependency check** (`dependency_scan.py`, pip-audit/safety)
  - Verifica vulnerabilità dipendenze
  - In CI: pip-audit, Gitleaks, Trivy

- **Hash check** (`hash_checker.py`)
  - Baseline SHA256 file; blocco se file critici modificati
  - Creazione baseline: `python -m security.hash_checker --build <root>`

- **Diff analysis** (`diff_analyzer.py`)
  - Modifiche sospette: auth/secret nel diff, endpoint nascosti, eval/exec aggiunti, commenti rimossi in zone sensibili

## 2. ANALISI STATICA (SAST)

- **Bandit** (Python): integrato in `sast_runner.py` e in pre-commit / `.pre-commit-config.yaml`
- **Semgrep**: configurabile in `.semgrep-rules.yml` (opzionale)
- Pattern custom in `advanced_verifier.py` e `malware_patterns.py`

## 3. DEPENDENCY VERIFICATION

- **pip-audit** / **safety**: `dependency_scan.py`, workflow `pre-push-security.yml`
- Verifica hash/registry: tramite pip; per firme GPG estendere script

## 4. CODE DIFF ANALYSIS

- `diff_analyzer.py`: analisi `git diff --cached`
- Red flag: modifiche a auth, endpoint nascosti, codice dopo return, eval/exec aggiunti

## 5. BLOCCO E ALERT

- **Hard block**: exit 1 se secret, malware, SAST HIGH, diff red flag, hash changed
- **Soft block**: `--soft` → exit 2, warning (es. dependency vuln)
- **Alert**: integrazione con `alerting.py` (webhook) per notifiche security team

## 6. COME USARE

**Installa hook Git (dalla root del repo):**
```bash
# Con Git Bash (Linux/mac/Windows con Git per Windows)
cp scripts/git_hooks/pre-commit .git/hooks/pre-commit
cp scripts/git_hooks/pre-push   .git/hooks/pre-push
chmod +x .git/hooks/pre-commit .git/hooks/pre-push
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_hooks.ps1
```

**Pre-commit framework (consigliato):**
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # esecuzione manuale
```

**Esecuzione manuale (senza hook):**
```bash
# Solo staged (come pre-commit)
python -m security.pre_commit_runner

# Full repo (come pre-push)
python -m security.pre_commit_runner --pre-push

# Con opzioni
python -m security.pre_commit_runner --skip-sast --skip-deps --soft --json
```

**Baseline hash (prima volta):**
```bash
python -m security.hash_checker --build .
```

## 7. CI/CD (GitHub Actions)

- **pre-push-security.yml**: su push/PR su main/master/develop
  - Runner FractalNova (pre_commit_runner --pre-push)
  - Gitleaks (secret)
  - Bandit, pip-audit, Trivy
  - Fail se `block_reasons` non vuoto

- **security.yml**: Ultra Security Guard (integrità, advanced verifier, backup) — già esistente

## 8. ESEMPIO INTEGRAZIONE (bash)

```bash
# .git/hooks/pre-commit (alternativa minima)
#!/bin/bash
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
python -m security.pre_commit_runner --root "$ROOT"
exit $?
```

Vedi anche `scripts/git_hooks/pre-commit` e `pre-push` per la versione completa con fallback a Bandit/safety se il modulo security non è disponibile.
