# Model Card — FractalNova

FractalNova è una **famiglia di modelli** per scrivere/rifinire/pubblicare libri.

## Modelli

| Modello | Tipo | Param | Lingue | Uso | Licenza |
|---|---|---|---|---|---|
| **FractalNova‑Pro** | Pretraining **da zero** | **~321B** | ✅ ~29 | Motore di punta: reasoning avanzato | **Proprietario** |
| **FractalNova‑Base** | Pretraining **da zero** | **~25B** | ✅ ~29 | Produzione: default (testo+copertine+export) | **Proprietario** |
| **FractalNova‑Mini** | Pretraining **da zero** | **~8B** | ✅ ~29 | Leggero, solo testo | **Apache-2.0 (open source)** |
| **FractalNova‑Core** | Pretraining **da zero** | ~124M | 1–2 | Ricerca / IP proprietario | **Proprietario** |

## FractalNova-Mini (8B) — Open Source

- **8B parametri** → funziona su **una GPU 12-16GB** (RTX 4070/4080/5060 Ti)
- **Apache-2.0**: pesi distribuiti pubblicamente, modifica e distribuzione libere
- **Solo testo**: generazione, umanizzazione, traduzione, analisi stile
- **NO copertine, NO pubblicazione, NO SEO visiva, NO export**
- Ideale per: sviluppatori, ricerca, uso locale leggero

## FractalNova-Base (25B) — Default Produzione

- **25B parametri** → su **una GPU 24GB** (RTX 4090 / RTX 5090)
- Pipeline completa: testo + copertine + SEO + pubblicazione + export
- **Proprietario**: non distribuibile

## FractalNova-Pro (321B) — Motore di Punta

- **321B parametri** → cluster multi-GPU
- Reasoning avanzato, capacità estese
- **Proprietario**: non distribuibile

## Dati di training

**Dataset sintetico** (10.000+ esempi) generato su 7 task × 6 lingue:
- write, continue, humanize, title, synopsis, seo, translate
- IT, EN, ES, FR, DE, PT

## Hardware consigliato

| Modello | GPU minima | VRAM | Open Source |
|---|---|---|---|
| **FractalNova-Mini (8B)** | RTX 4070 / RTX 5060 Ti | 12–16GB | ✅ Apache-2.0 |
| **FractalNova-Base (25B)** | RTX 4090 / RTX 5090 | 24GB | ❌ Proprietario |
| **FractalNova-Pro (321B)** | Cluster multi-GPU | 128GB+ | ❌ Proprietario |
| **FractalNova-Core (124M)** | RTX 5060 Ti | 16GB | ❌ Proprietario |

## Licenza
- **Mini (8B)**: Apache-2.0 (open source)
- **Pro, Base, Core**: LICENSE-MODEL (proprietario, non distribuibile)
- Codice: LICENSE-CODE
