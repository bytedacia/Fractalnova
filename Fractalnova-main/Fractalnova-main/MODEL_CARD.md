# Model Card — FractalNova

FractalNova è una **famiglia di modelli proprietari** per scrivere/rifinire/pubblicare libri.

| Modello | Tipo | Param | Lingue | Uso | Licenza |
|---|---|---|---|---|---|
| **FractalNova‑Pro** | Pretraining **da zero** | **~321B** | ✅ ~29 | Motore di punta: lunghe capacità, reasoning avanzato | **Proprietario** |
| **FractalNova‑Base** | Pretraining **da zero** | **~25B** | ✅ ~29 | Produzione: scrittura, riscrittura, SEO, traduzione (default) | **Proprietario** |
| **FractalNova‑Core** | Pretraining **da zero** | ~124M | 1–2 | Ricerca / IP proprietario | **Proprietario** |

> ⚠️ **Tutti i modelli FractalNova sono proprietari.** Nessun peso è distribuito pubblicamente.
> L'accesso è soggetto a licenza proprietaria (vedi `LICENSE-MODEL`).

## Perché FractalNova-Base (25B) è il default

- **25B parametri** → bilancia potenza e consumi
- Funziona su **una singola GPU 24GB** (es. RTX 4090 / RTX 5090)
- **~29 lingue** con qualità "umana" nelle principali
- Training efficiente: pretraining da zero senza dipendenze da modelli esterni
- **Proprietario**: nessuna dipendenza da API di terze parti, totale controllo sui dati

## Dati di training

**Dataset sintetico** (10.000+ esempi) generato su 7 task × 6 lingue:
- write, continue, humanize, title, synopsis, seo, translate
- IT, EN, ES, FR, DE, PT

**Produzione**: continuous training loop da feedback utenti (`train_continuous.py`).

## Procedura di training

1. **Dataset generation**: 10k+ esempi sintetici
2. **SFT (QLoRA)**: 4‑bit NF4, LoRA r=32 α=64, seq_len=2048, batch=16
3. **DPO Alignment**: preference pairs con LLM-as-judge, β=0.1
4. **Merge**: adapter → modello standalone per vLLM

## Hardware consigliato

| Modello | GPU minima | VRAM | Note |
|---|---|---|---|
| **FractalNova-Base (25B)** | RTX 4090 / RTX 5090 | 24GB | Default produzione |
| **FractalNova-Pro (321B)** | Cluster multi-GPU | 128GB+ | Motore di punta |
| **FractalNova-Core (124M)** | RTX 5060 Ti | 16GB | Ricerca / testing |

## Valutazione

```bash
python training/benchmark.py --models fractalnova-base,fractalnova-pro,fractalnova-core
```

## Limiti (onesti)
- FractalNova-Core (~124M) è piccolo: utile come IP, non come produttore primario
- Possibili allucinazioni su fatti specifici: revisione umana sempre raccomandata
- Modelli proprietari: nessun accesso pubblico ai pesi

## Licenza
Codice: LICENSE-CODE. Pesi: LICENSE-MODEL (proprietario, non distribuibile).
