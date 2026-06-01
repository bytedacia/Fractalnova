# FractalNova · Model Training

Costruisci il **tuo modello di scrittura** partendo da Qwen3-4B (fine-tuning QLoRA)
oppure da zero con FractalNova-Core (124M). Tutto su **una GPU consumer 16GB**.

## Le due tracce

| Traccia | Base | Param | Lingue | Addestramento |
|---|---|---|---|---|
| **FractalNova-Pro** | Qwen3-4B (fine-tune QLoRA) | ~4B | ~29 | Adapter LoRA in ~8-10 GB VRAM |
| **FractalNova-Core** | Da zero (tuo IP) | ~124M | 1-2 | Pretraining in <16 GB |

## Setup

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -r training/requirements-train.txt
```

## Traccia Pro — fine-tuning Qwen3-4B

```bash
# 1) Genera 10.000+ esempi sintetici (richiede GOOGLE_API_KEY)
python training/dataset_generator.py --generate-all --num-per-task 500 --languages it,en,es,fr,de

# 2) Prepara dataset in formato chat
python training/prepare_dataset.py \
    --inputs training/data/generated/all_generated.jsonl \
    --out-dir training/data

# 3) (Opzionale) Hyperparameter tuning con Optuna
python training/hyperparameter_tuning.py --n-trials 30

# 4) Supervised Fine-Tuning (QLoRA)
python training/train_qlora.py --config training/configs/qlora_5060ti.yaml

# 5) Genera preference pairs per DPO
python training/train_dpo.py --generate-pairs --base-data training/data/train.jsonl --max-pairs 2000

# 6) DPO Alignment
python training/train_dpo.py --base-model Qwen/Qwen3-4B \
    --adapter training/outputs/fractalnova-qlora \
    --output training/outputs/fractalnova-dpo

# 7) Valuta
python training/evaluate.py --base Qwen/Qwen3-4B \
    --adapter training/outputs/fractalnova-qlora --load-4bit

# 8) Benchmark contro GPT-4 / Claude / Gemini
python training/benchmark.py --models gemini-2.0-flash,fractalnova-pro,base-qwen

# 9) Merge per deploy
python training/merge_and_export.py \
    --base Qwen/Qwen3-4B \
    --adapter training/outputs/fractalnova-dpo \
    --out training/outputs/fractalnova-pro-merged
```

Oppure tutto in un colpo solo:

```bash
python training/pipeline.py --full-cycle
```

## Traccia Core — pretraining da zero

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
├── benchmark.py                      # Benchmark vs GPT-4/Claude/Gemini
├── pipeline.py                       # Orchestratore end-to-end
├── evaluate.py / infer.py            # Valutazione e inferenza
├── merge_and_export.py               # Merge adapter → modello standalone
├── _common.py                        # Loader condiviso
└── pretrain/                         # Modello da zero (FractalNova-Core)
```

## Strategia

Qwen3-4B è la scelta ottimale perché:
- **4B parametri** vs 7B = 43% meno VRAM, inferenza 2x più veloce
- **Batch size 2** vs 1 (stessa VRAM) = training 2x più rapido
- **Max_seq_len 2048** vs 1024 = capitoli più coerenti
- **Costo API** più basso in produzione
- Abbastanza parametri per specializzarsi sul dominio libri
