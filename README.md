# FractalNova

**FractalNova** è il sistema unificato per generare, rifinire e pubblicare libri con intelligenza artificiale, in locale.

**Repository:** [https://github.com/bytedacia/scribenova](https://github.com/bytedacia/scribenova)  
Clone: `git clone https://github.com/bytedacia/scribenova.git`

## Panoramica
- Generazione principale con DeepSeek‑V3 (capitoli lunghi, nessun limite pratico di caratteri)
- Umanizzazione/correzione con Qwen3
- Titolo e sinossi con Llama 3 (poi umanizzati da Qwen3)
- SEO con Gemma/gemini (keywords, meta, categorie)
- Copertina con FLUX (diffusers)
- Export Word e Wattpad
- Ricerca editori (Google CSE) ed email di proposta con allegati
- pubblicazione on amazon kindle publishing 

## Modelli utilizzati
- DeepSeek‑V3 (generatore principale) — [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3.git)
- Qwen3 — [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3.git)
- Llama 3 — [meta-llama/llama3](https://github.com/meta-llama/llama3.git)
- Gemma — [google-deepmind/gemma](https://github.com/google-deepmind/gemma.git)
- FLUX — [black-forest-labs/flux](https://github.com/black-forest-labs/flux.git)

Tutti i pesi sono scaricati da Hugging Face al primo avvio o via script.

## Requisiti
- Python 3.9+
- GPU NVIDIA consigliata (CUDA) per performance e qualità
- Dipendenze principali: `torch`, `transformers`, `diffusers`, `accelerate`, `huggingface_hub`, `flask`, `python-docx`, `beautifulsoup4`, `requests`, `python-dotenv`

### Requisiti hardware (per esecuzione locale)

GPU supportate

- NVIDIA RTX 50 Series
  - RTX 5090 (Prestazioni ottimali)
  - RTX 5080  (ottime prestazioni)
  - RTX 5070   (medie prestazioni)
  - RTX 5070TI
- NVIDIA RTX 40 Series
  - RTX 4090
  - RTX 4080 Super
  - RTX 4080
  - RTX 4070 Ti Super
  - RTX 4070 Ti
  - RTX 4070 Super
  - RTX 4070
- NVIDIA RTX 30 Series
  - RTX 3090 Ti
  - RTX 3090
  - RTX 3080 Ti
  - RTX 3080
  - RTX 3070 Ti
  - RTX 3070
- AMD e Apple Silicon
  - AMD RX 7900 XTX
  - AMD RX 7900 XT
  - Apple M3 Max / M3 Pro
  - Apple M2 Ultra

GPU con supporto limitato

- RTX 4060 Ti (supportata ma con prestazioni ridotte)
- RTX 3060 Ti (supportata ma con prestazioni ridotte)

GPU non supportate

- NVIDIA serie x50/x60: RTX 5060, RTX 5050, RTX 4060, RTX 4050, RTX 3060, RTX 3050
- Serie RTX 20 e precedenti (RTX 2080 Ti e inferiori)
- Tutte le GPU GTX e GPU integrate Intel

CPU consigliate

- Intel 12th Gen e successive (consigliata)
  - i9-13900K/KF, i9-12900K/KF, i7-13700K/KF, i7-12700K/KF
- Intel 11th Gen (minimo)
  - i9-11900K, i7-11700K
- AMD Ryzen 7000 (consigliata)
  - 7950X, 7900X, 7800X3D, 7700X
- AMD Ryzen 5000 (minimo)
  - 5950X, 5900X, 5800X3D, 5800X
- Apple Silicon
  - Serie M3 (consigliata): M3 Max/Pro/M3
  - Serie M2 (supportata): M2 Ultra/Max/Pro/M2

Memoria

- DDR5: minimo DDR5-5200, consigliato DDR5-6000+

## Installazione
```bash
git clone https://github.com/bytedacia/scribenova.git
cd scribenova
python -m venv venv
./venv/Scripts/activate  # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Facoltativo: token Hugging Face se richiesto dai repo (modelli soggetti ad accesso):
```bash
setx HF_TOKEN "hf_xxx"  # Windows
# export HF_TOKEN=hf_xxx   # Linux/macOS
```

Scarico modelli (opzionale, altrimenti auto-download al primo avvio):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

## Variabili ambiente utili
- Autore/Email: `AUTHOR_NAME`, `AUTHOR_EMAIL`
- SMTP: `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`
- Google CSE: `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_CX`
- Modelli (override opzionali):
  - `DEEPSEEK_LOCAL_PATH`, `DEEPSEEK_REPO_ID`, `DEEPSEEK_REVISION`, `DEEPSEEK_MAX_NEW_TOKENS` (default 4096)
  - `QWEN_LOCAL_MODEL_PATH`, `QWEN_REPO_ID`, `QWEN_REVISION`
  - `LLAMA_LOCAL_MODEL_PATH`, `LLAMA_REPO_ID`, `LLAMA_REVISION`
  - `GEMMA_LOCAL_MODEL_PATH`, `GEMMA_REPO_ID`, `GEMMA_REVISION`
  - `FLUX_MODEL_ID`, `FLUX_LOCAL_MODEL_PATH`, `FLUX_REPO_ID`, `FLUX_REVISION`
- Hugging Face token: `HF_TOKEN` o `HUGGINGFACE_HUB_TOKEN`
- **FractalNova UI (Gradio):** `FRACTALNOVA_PORT` (default 7860), `FRACTALNOVA_HOST` (default 0.0.0.0)
- **DeepSeek (app Gradio):** `DEEPSEEK_MODEL_PATH`, `DEEPSEEK_CONFIG_PATH` (fallback: `DEESEEK_*`)

### Sicurezza (consigliato)
- `API_SECRET`: abilita autenticazione via header `X-API-KEY`
- `CORS_ORIGINS`: lista di origin consentiti, separati da virgola
- `MAX_CONTENT_LENGTH_MB`: limite dimensione richiesta (default 10)
- `RATE_LIMIT` e `RATE_WINDOW_SEC`: rate limiting per IP (default 30 richieste/10 min)
- `ALLOW_TRUST_REMOTE_CODE`: per sicurezza è disattivato; abilitalo solo se necessario

## Avvio rapido (orchestratore unico)
```python
from inference.orchestrator import FractalNova

us = FractalNova()  # scarica i modelli se mancanti
result = us.run({
    "title": "",
    "genre": "fiction",
    "chapter_structure": "misto",
    "target_pages": 500
})
print(result["book_structure"]["title"], result["word_path"], result["cover_path"]) 
```

## Interfaccia web (Flask)
```bash
python inference/generate.py
# apri http://localhost:5000
```

## Note
- DeepSeek‑V3 è il modello principale per la generazione dei capitoli. Gli altri modelli sono perfezionatori.
- Su CPU funziona, ma è consigliata una GPU per tempi ragionevoli.
- Evita di fare commit dei pesi: la cartella `models/` è ignorata nel repo.

## 🛡️ Sistema di Sicurezza Ultra-Avanzato (24/7)

**Pipeline completa (assessment, hardening, auth, backup, IA, monitoraggio, DR):** vedi [security/SECURITY_PIPELINE.md](security/SECURITY_PIPELINE.md). Per operazioni e incident response: [security/RUNBOOK.md](security/RUNBOOK.md).

### Protezione Multi-Livello
Il sistema include **4 livelli di sicurezza** che controllano TUTTO il codice 24/7, inclusi i file di sicurezza stessi:

#### 🔍 **Livello 1: Advanced Security Verifier**
- **File**: `security/advanced_verifier.py`
- **Funzione**: Scansione ultra-aggressiva per codice malevolo
- **Pattern rilevati**:
  - Esecuzione pericolosa (`eval`, `exec`, `compile`, `__import__`)
  - Accesso sistema (`os.system`, `subprocess`, `shell=True`)
  - Manipolazione file (`remove`, `unlink`, `rmtree`)
  - Rete sospetta (`requests` con `verify=False`, `socket`)
  - Serializzazione pericolosa (`pickle.loads`, `marshal`)
  - **Auto-modifica** (`open(__file__)`, `exec(open(__file__))`)
  - Bypass sicurezza (`# bypass`, `# hack`, `# backdoor`)
  - Obfuscazione (`chr()`, `lambda`, `getattr`)

#### 🔒 **Livello 2: Self-Monitor (Auto-Protezione)**
- **File**: `security/self_monitor.py`
- **Funzione**: Auto-monitoraggio dei file di sicurezza
- **Caratteristiche**:
  - Monitora TUTTI i file critici inclusi quelli di sicurezza
  - Rileva auto-modifica in tempo reale
  - Baseline automatica con hash SHA256
  - Allerte immediate per modifiche ai file di sicurezza
  - Risposta di emergenza automatica

#### 🚨 **Livello 3: Ultra Guard (Orchestratore)**
- **File**: `security/ultra_guard.py`
- **Funzione**: Coordinamento di tutti i sistemi di sicurezza
- **Caratteristiche**:
  - Monitoraggio continuo 24/7
  - Risposta di emergenza automatica
  - Criptazione automatica file critici
  - Backup di emergenza
  - Notifiche di sicurezza
  - Auto-riparazione

#### 🔐 **Livello 4: Encryption & Protection**
- **File**: `security/encryptor.py`
- **Funzione**: Criptazione file critici con Fernet
- **Caratteristiche**:
  - Criptazione automatica su threat
  - Chiavi di sicurezza configurabili
  - Decriptazione sicura

### 🚀 **Avvio Sistema di Sicurezza**

#### **Metodo 1: Script Interattivo**
```bash
# Avvia sistema completo
python security/start_ultra_security.py

# Comandi disponibili:
#   status    - Stato sistema
#   baseline  - Crea baseline
#   scan      - Scansione una volta
#   start     - Protezione continua 24/7
```

#### **Metodo 2: Comandi Diretti**
```bash
# Crea baseline di sicurezza
python security/self_monitor.py . create_baseline

# Verifica integrità
python security/self_monitor.py . check

# Scansione avanzata
python security/advanced_verifier.py .

# Avvia protezione continua
python security/ultra_guard.py . start
```

#### **Metodo 3: Scansione di Emergenza**
```bash
# Scansione immediata ultra-avanzata
python security/ultra_guard.py . emergency_scan
```

### 🔄 **GitHub Actions - Protezione 24/7**

Il sistema è integrato con GitHub Actions per monitoraggio continuo:

- **Frequenza**: Ogni 15 minuti
- **Trigger**: Push, PR, schedule
- **Azioni**:
  - Verifica integrità file
  - Scansione avanzata codice malevolo
  - Scansione di emergenza
  - Upload report di sicurezza
  - **Build fallisce** su threat critici

### ⚙️ **Configurazione Avanzata**

#### **Variabili d'Ambiente**
```bash
# Chiave di criptazione (opzionale)
SECURITY_ENC_KEY=your_encryption_key_here

# Configurazione monitoraggio
SECURITY_SCAN_INTERVAL=60          # Secondi tra scansioni
SECURITY_EMERGENCY_RESPONSE=true   # Risposta emergenza
SECURITY_AUTO_ENCRYPT=true         # Criptazione automatica
```

#### **File di Configurazione**
- `security/ultra_guard_config.json` - Configurazione sistema
- `security/self_monitor_baseline.json` - Baseline file
- `security/ultra_guard_status.json` - Stato sistema
- `security/security_alerts.json` - Allerte di sicurezza

### 📊 **Monitoraggio e Report**

#### **File di Log**
- `security/ultra_guard_activity.log` - Log attività
- `security/emergency_log.json` - Log emergenze
- `security/advanced_security_report.json` - Report scansioni

#### **Statistiche Sistema**
- Scansioni eseguite
- Threat bloccati
- Risposte di emergenza
- File protetti
- Ultima scansione

### 🔧 **Sistema di Auto-Riparazione**

#### **Backup Sicuro Automatico**
- **File**: `security/backup_manager.py`
- **Funzione**: Backup criptato di tutti i file critici
- **Caratteristiche**:
  - Backup automatico ogni ora
  - Criptazione con Fernet
  - Hash SHA256 per verifica integrità
  - Rotazione backup (max 10 per file)
  - Pulizia automatica backup vecchi

#### **Auto-Riparazione**
- **File**: `security/auto_repair.py`
- **Funzione**: Rileva e ripara automaticamente file infetti
- **Caratteristiche**:
  - Rilevamento infezioni in tempo reale
  - Quarantena file infetti
  - Ripristino da backup puliti
  - Verifica post-riparazione
  - Riparazione di emergenza

#### **Processo di Auto-Riparazione**
1. **Rilevamento**: Scansiona file per pattern malevoli
2. **Quarantena**: Sposta file infetti in quarantena sicura
3. **Ripristino**: Decripta e ripristina file puliti da backup
4. **Verifica**: Controlla integrità file riparato
5. **Log**: Registra tutte le operazioni di riparazione

### 🚨 **Risposta di Emergenza**

Quando rileva threat critici, il sistema:

1. **Cripta automaticamente** file critici
2. **Crea backup** di emergenza
3. **Avvia auto-riparazione** immediata
4. **Quarantena** file infetti
5. **Ripristina** file puliti da backup criptati
6. **Verifica** integrità sistema
7. **Invia notifiche** di sicurezza
8. **Genera report** dettagliati

### 🔧 **Manutenzione**

```bash
# Verifica stato sistema
python security/ultra_guard.py . status

# Aggiorna baseline
python security/self_monitor.py . create_baseline

# Scansione manuale
python security/advanced_verifier.py .

# Backup sicuro
python security/backup_manager.py . backup

# Auto-riparazione
python security/auto_repair.py . scan

# Riparazione di emergenza
python security/auto_repair.py . emergency

# Stato quarantena
python security/auto_repair.py . quarantine
```

### ⚠️ **Note Importanti**

- Il sistema monitora **TUTTI i file** inclusi quelli di sicurezza
- **Auto-protezione** previene modifiche malevoli ai file di sicurezza
- **Criptazione automatica** su threat critici
- **Monitoraggio 24/7** con GitHub Actions
- **Risposta immediata** a minacce

## Licenza
MIT (codice). L’uso dei modelli segue le rispettive licenze.

## Ringraziamenti
- DeepSeek‑V3 — generazione long‑form
- Qwen3 — umanizzazione/proof
- Llama 3 — titolo e sinossi
- Gemma — SEO
- FLUX — copertina
