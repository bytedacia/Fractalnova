<div align="center">

# FractalNova

**Sistema multilingua per generare, rifinire e pubblicare libri con l'intelligenza artificiale — in locale.**

[![CI](https://github.com/bytedacia/scribenova/actions/workflows/ci.yml/badge.svg)](https://github.com/bytedacia/scribenova/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-vedi%20LICENSE-lightgrey)

</div>

> **Repository:** https://github.com/bytedacia/scribenova
> `git clone https://github.com/bytedacia/scribenova.git`

FractalNova combina una **famiglia di modelli proprietari** con una **pipeline editoriale completa**:
dalla scrittura del manoscritto ai file pronti per la pubblicazione (EPUB/PDF/Word) e ai metadati per gli store.

---

## ✨ Cosa fa

| Funzione | Modulo | Stato |
|---|---|---|
| Generazione capitoli (long-form) | `inference/` (DeepSeek-V3) | ✅ (richiede pesi+GPU) |
| Umanizzazione / riscrittura | `inference/` (Qwen3) | ✅ (richiede pesi) |
| Titolo & sinossi | `inference/` (Llama 3) | ✅ (richiede pesi) |
| SEO (keyword, meta, categorie) | `fractalnova/seo.py` | ✅ parsing robusto + test |
| Copertina | `inference/` (FLUX/diffusers) | ✅ (richiede GPU) |
| Export **Word · Wattpad · EPUB · PDF** | `fractalnova/export.py` | ✅ EPUB/Wattpad puro Python + test |
| Pacchetto pubblicazione & metadati **KDP** | `fractalnova/publishing.py` | ✅ con **consenso** (no invii automatici) |
| Modello proprietario **da zero** + **fine-tune** | `training/` | ✅ codice completo |

> Gli step modello degradano con grazia: senza pesi/GPU la pipeline **non crasha**, così puoi
> validare struttura ed export anche su una macchina senza CUDA.

---

## 🧠 I modelli (strategia a quattro tracce)

| | Tipo | Parametri | Multilingua "umano" | Ruolo | Licenza |
|---|---|---|---|---|---|
| **FractalNova-Pro** | Pretraining **da zero** | **~321B** | ✅ ~29 lingue | Motore di punta | **Proprietario** |
| **FractalNova-Base** | Pretraining **da zero** | **~25B** | ✅ ~29 lingue | Motore di produzione | **Proprietario** |
| **FractalNova-Mini** | Pretraining **da zero** | **~8B** | ✅ ~29 lingue | Leggero, test-only, open source | **Apache-2.0** |
| **FractalNova-Core** | Pretraining **da zero** | ~124M | ⚠️ 1–2 lingue | Nucleo proprietario / ricerca | **Proprietario** |

> ⚠️ **Pro, Base e Core sono modelli proprietari.** I pesi non sono distribuiti pubblicamente.
> L'accesso è soggetto a licenza proprietaria (vedi [`LICENSE-MODEL`](LICENSE-MODEL)).
>
> **FractalNova-Mini (8B)** è **open source** (Apache-2.0). Genera solo testo:
> non include copertine, pubblicazione, SEO visiva né export. Peso leggero,
> ideale per uso locale su GPU consumer.

### FractalNova-Mini — cosa include e cosa NO

| | Mini (8B) | Base/Pro |
|---|---|---|
| Generazione testo | ✅ | ✅ |
| Umanizzazione | ✅ | ✅ |
| Traduzione | ✅ | ✅ |
| Analisi stile | ✅ | ✅ |
| Copertine (FLUX) | ❌ | ✅ |
| SEO visiva / visione | ❌ | ✅ |
| Pubblicazione KDP | ❌ | ✅ |
| Export EPUB/DOCX/PDF | ❌ | ✅ |
| Open source | ✅ Apache-2.0 | ❌ Proprietario |

Dettagli, comandi e note hardware: [`training/README.md`](training/README.md) · scheda: [`MODEL_CARD.md`](MODEL_CARD.md).

## Architettura · Una sola IA

```python
from inference.fractalnova import FractalNovaInference

ai = FractalNovaInference("pro")    # FractalNova-Pro 321B — modello proprietario (no pesi pubblici)
ai = FractalNovaInference("base")   # FractalNova-Base 25B — modello proprietario (default produzione)
ai = FractalNovaInference("mini")   # FractalNova-Mini 8B — open source (Apache-2.0), solo testo
ai = FractalNovaInference("core")   # 124M da zero (IP proprietario)

ai.generate("Scrivi un racconto...")                       # testo
ai.humanize(testo)                                          # umanizzazione
ai.translate(testo, "en")                                   # traduzione
ai.analyze_style(testo)                                     # stile
ai.analyze_seo(libro)                                       # SEO
ai.analyze_image_seo("cover.png")                           # SEO visiva
ai.describe_image("cover.png")                              # descrizione
ai.classify_genre_from_cover("cover.png")                   # genere
ai.generate_cover("Titolo", "fantasy")                      # copertina
ai.generate_book({"title":"...", "genre":"..."})            # libro
ai.run({"title":"...", "genre":"..."})                      # pipeline
```

| Modulo | Descrizione | Modello |
|---|---|---|
| Testo / Chat | generazione, stile, traduzione | **FractalNova-Mini 8B** (open source) / **Base 25B** (default) / **Pro 321B** (premium) |
| Visione | analisi copertine, SEO visiva, genere | **Gemma-4-E2B** (solo Base/Pro) |
| Copertina | generazione immagini | **FLUX.1-dev** (solo Base/Pro) |
| Pubblicazione | EPUB, DOCX, PDF, Wattpad, KDP | `fractalnova/export.py` (solo Base/Pro) |
| Addestramento | QLoRA + DPO + pretraining | `training/` |
| Frontend | Gradio 4 tab (`app.py`) | testi, stile, libri, copertine |

> **Onestà:** FractalNova-Mini (8B) è l'unico modello **open source** (Apache-20). Genera solo testo.
> FractalNova-Base (25B) e Pro (321B) sono **proprietary** e includono l'intera pipeline completa.

---

## 🚀 Quickstart (prodotto)

```bash
git clone https://github.com/bytedacia/scribenova.git && cd scribenova
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env        # inserisci GOOGLE_API_KEY e i path modelli

# Genera un libro end-to-end (scrittura → SEO → copertina → EPUB/PDF/Word → pacchetto KDP)
python fractalnova_cli.py --title "Il Sale degli Dei" --genre fantasy --pages 120
```

Interfaccia web (Gradio): `python app.py` → http://localhost:7860

## 🏋️ Quickstart (training del tuo modello)

```bash
pip install -e ".[train]"
# Genera dataset (10k+ esempi, richiede GOOGLE_API_KEY)
python training/dataset_generator.py --generate-all --num-per-task 500
# Prepara e addestra
python training/prepare_dataset.py --inputs training/data/generated/all_generated.jsonl --out-dir training/data
python training/train_qlora.py --config training/configs/qlora_5060ti.yaml
# DPO alignment (opzionale ma raccomandato)
python training/train_dpo.py --generate-pairs --base-data training/data/train.jsonl
python training/train_dpo.py --base-model Qwen/Qwen3-4B --adapter training/outputs/fractalnova-qlora
# Merge per deploy
python training/merge_and_export.py --base Qwen/Qwen3-4B --adapter training/outputs/fractalnova-dpo --out training/outputs/fractalnova-pro-merged
# poi in .env:  QWEN_LOCAL_MODEL_PATH=training/outputs/fractalnova-pro-merged
```

---

## 🗂️ Struttura

```
fractalnova/        Pacchetto di produzione: config, logging, SEO, export, publishing
inference/          Modello DeepSeek-V3 + orchestratore + funzioni pipeline + app Flask
training/           Modello proprietario: Core (da zero) e Pro (fine-tune QLoRA)
tests/              Test (girano anche senza GPU)
app.py              UI Gradio        |  fractalnova_cli.py  CLI end-to-end
.env.example        Tutte le variabili di configurazione
```

## 🧪 Qualità

```bash
pip install -e ".[dev]"
pytest && ruff check fractalnova tests fractalnova_cli.py
```

## 🔒 Sicurezza & etica
- Suite di sicurezza 24/7 (assessment, hardening, monitoraggio, DR): vedi
  [security/SECURITY_PIPELINE.md](security/SECURITY_PIPELINE.md) e [security/RUNBOOK.md](security/RUNBOOK.md).
- Outreach agli editori **opt-in** con revisione umana (niente spam automatico).
- Nessuna iniezione di errori nel testo; nessun prompt "senza limiti".
- Segreti solo in `.env` (mai committati).

## 📦 Requisiti hardware
GPU NVIDIA consigliata per l'inferenza dei modelli pesanti. Per il fine-tuning QLoRA basta
**una GPU da 16GB** (es. RTX 5060 Ti — nota Blackwell/CUDA 12.8 in [`training/README.md`](training/README.md)).
Tabella estesa GPU/CPU/RAM: vedi sezione hardware in fondo a questo file nella storia del repo.

## 📄 Licenza
- **Codice**: [`LICENSE-CODE`](LICENSE-CODE) — licenza open source.
- **Modelli proprietari**: [`LICENSE-MODEL`](LICENSE-MODEL) — FractalNova-Pro (321B), FractalNova-Base (25B), FractalNova-Core (124M). **Non distribuibili.**
- **FractalNova-Mini (8B)**: **Apache-2.0** — open source, pesi distribuiti pubblicamente.
- Rispettare le licenze dei modelli di terze parti (es. FLUX.1-dev → Black Forest Labs).

> ⚠️ I pesi di Pro, Base e Core non sono inclusi in questo repository e non sono distribuiti.
> FractalNova-Mini (8B) è disponibile come modello open source su Hugging Face.

## 📝 Citazione
Vedi [`CITATION.cff`](CITATION.cff).

## 🙏 Ringraziamenti
DeepSeek-V3 (long-form) · Qwen (umanizzazione/multilingua) · Llama 3 (titolo/sinossi) · Gemma (SEO) · FLUX (copertina).
