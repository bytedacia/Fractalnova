# FractalNova · Model Training

Costruisci i modelli FractalNova: **Base (25B)**, **Pro (321B)** o **Core (124M)** — tutti proprietari, tutti da zero.

## Le tre tracce

| Traccia | Tipo | Param | Lingue | Hardware | Ruolo |
|---|---|---|---|---|---|
| **FractalNova-Base** | Pretraining da zero | **~25B** | ✅ ~29 | 1× GPU 24GB (RTX 4090/5090) | **Default produzione** |
| **FractalNova-Pro** | Pretraining da zero | **~321B** | ✅ ~29 | Cluster multi-GPU | Motore di punta |
| **FractalNova-Core** | Pretraining da zero | ~124M | 1–2 | 1× GPU 16GB (RTX 5060 Ti) | Ricerca / testing |

> ⚠️ **Tutti i modelli sono proprietari.** Nessun peso è distribuito pubblicamente.

## Setup

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -r training/requirements-train.txt
```

## Traccia Base — FractalNova-Base (25B) — DEFAULT

```bash
# 1) Genera 10.000+ esempi sintetici
python training/dataset_generator.py --generate-all --num-per-task 500 --languages it,en,es,fr,de,pt

# 2) Prepara dataset
python training/prepare_dataset.py \
    --inputs training/data/generated/all_generated.jsonl \
    --out-dir training/data

# 3) Pretraining da zero
python training/pretrain/pretrain.py --config training/pretrain/configs/fractalnova_base_25b.yaml

# 4) Fine-tuning su dominio libri
python training/train_qlora.py --config training/configs/qlora_base_25b.yaml

# 5) DPO Alignment
python training/train_dpo.py --generate-pairs --base-data training/data/train.jsonl --max-pairs 5000
python training/train_dpo.py --base-model fractalnova-base-25b \
    --adapter training/outputs/fractalnova-base-qlora \
    --output training/outputs/fractalnova-base-dpo

# 6) Valuta e benchmark
python training/evaluate.py --base fractalnova-base-25b --load-4bit
python training/benchmark.py --models fractalnova-base,fractalnova-pro,fractalnova-core
```

## Traccia Pro — FractalNova-Pro (321B)

```bash
# Pretraining da zero (richiede cluster multi-GPU)
python training/pretrain/pretrain.py --config training/pretrain/configs/fractalnova_pro_321b.yaml

# Fine-tuning + DPO (stessa procedura di Base, parametri più grandi)
python training/train_fractalnova_pro.py --config training/configs/pro_321b.yaml
```

## Traccia Core — FractalNova-Core (124M)

```bash
python training/pretrain/tokenizer_train.py --input training/data/corpus --out training/pretrain/artifacts
python training/pretrain/prepare_corpus.py --input training/data/corpus --artifacts training/pretrain/artifacts
python training/pretrain/pretrain.py --config training/pretrain/configs/fractalnova_core_124m.yaml
python training/pretrain/sample.py --ckpt training/pretrain/outputs/core-124m/ckpt.pt \
    --tokenizer training/pretrain/artifacts/tokenizer.json --prompt "C'era una volta"
```

## Continuous Training (produzione)

```bash
# Raccogli dati da produzione + riaddestra
python training/train_continuous.py --pipeline

# O via cron:
# 0 3 * * 1 cd /app && python training/train_continuous.py --pipeline
```

## Struttura

```
training/
├── configs/                          # Config SFT + DPO + Pipeline
├── data/                             # Dataset (sample + generated)
├── dataset_generator.py              # Genera 10k+ esempi sintetici
├── prepare_dataset.py                # Raw → chat format
├── train_qlora.py                    # QLoRA fine-tuning
├── train_dpo.py                      # DPO alignment
├── hyperparameter_tuning.py          # Optuna
├── train_continuous.py               # Continuous training
├── train_fractalnova_pro.py          # Training FractalNova-Pro 321B
├── benchmark.py                      # Benchmark vs modelli
├── pipeline.py                       # Orchestratore end-to-end
├── evaluate.py / infer.py            # Valutazione e inferenza
├── merge_and_export.py               # Merge adapter → modello standalone
├── _common.py                        # Loader condiviso
└── pretrain/                         # Pretraining da zero
    ├── configs/
    │   ├── fractalnova_base_25b.yaml # Config FractalNova-Base (25B)
    │   ├── fractalnova_pro_321b.yaml # Config FractalNova-Pro (321B)
    │   └── fractalnova_core_124m.yaml # Config FractalNova-Core (124M)
    └── ...
```

## Strategia

FractalNova-Base (25B) è il default perché:
- **25B parametri** → potenza sufficiente per specializzazione profonda sul dominio libri
- **~29 lingue** con qualità "umano" nelle principali
- **Proprietario da zero**: nessuna dipendenza da modelli di terze parti
- Funziona su **1× GPU 24GB** (RTX 4090 / RTX 5090)
- **Training efficiente**: pretraining da zero con dati proprietari
