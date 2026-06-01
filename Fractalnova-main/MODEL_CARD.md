# Model Card — FractalNova

FractalNova è una **famiglia a due modelli** per scrivere/rifinire/pubblicare libri.

| Modello | Tipo | Param | Lingue | Uso |
|---|---|---|---|---|
| **FractalNova‑Pro** | Fine‑tune QLoRA 4‑bit di `Qwen3‑4B` | ~4B | ~29 | Produzione: scrittura, riscrittura, titolo, sinossi, SEO, traduzione |
| **FractalNova‑Core** | Pretraining **da zero** | ~124M | 1–2 | Ricerca / IP proprietario |

## Perché Qwen3-4B invece di 7B
- 43% meno VRAM → batch size 2x, max_seq_len 2048
- Inferenza 2x più veloce, costo API dimezzato
- Abbastanza parametri per specializzarsi sul dominio libri
- Training più rapido (più iterazioni nello stesso tempo)

## Dati di training

**Dataset sintetico** (10.000+ esempi) generato via Gemini API su 7 task × 6 lingue:
- write, continue, humanize, title, synopsis, seo, translate
- IT, EN, ES, FR, DE, PT
- Generato da `training/dataset_generator.py`

**Produzione**: continuous training loop da feedback utenti (`train_continuous.py`).

## Procedura di training

1. **Dataset generation**: Gemini API → 10k+ esempi sintetici
2. **SFT (QLoRA)**: 4‑bit NF4, LoRA r=32 α=64, seq_len=2048, batch=16
3. **DPO Alignment**: preference pairs con LLM-as-judge, β=0.1
4. **Merge**: adapter → modello standalone per vLLM

Hardware: **1× GPU 16GB** (RTX 5060 Ti). Tempo totale: ~8h.

## Valutazione

```bash
python training/benchmark.py --models gemini-2.0-flash,fractalnova-pro,base-qwen
```

## Limiti (onesti)
- Non è un modello generalista di frontiera (GPT-4/Claude/Gemini = 100B+)
- Core è piccolo (~124M): utile come IP, non come produttore primario
- Possibili allucinazioni su fatti specifici: revisione umana sempre raccomandata

## Licenza
Codice: LICENSE-CODE. Pesi: LICENSE-MODEL + Apache-2.0 (Qwen3).
