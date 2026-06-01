"""
FractalNova · FULL fine-tuning / continued-pretraining (SUL SERVER GRANDE).

Addestra TUTTI i pesi del modello (non solo un adapter): i pesi diventano NUOVI
(numericamente distinti dalla base), il comportamento/voce/identita' sono FractalNova.
Questo e' il massimo realistico per avere "pesi nuovi" restando coerente.

VERITA': cosi' i pesi sono nuovi e il modello NON si comporta come Qwen e si
identifica come FractalNova. Resta tracciabile solo l'ARCHITETTURA (config qwen3:
hidden_size, num_layers, vocab) -> per cancellare anche quella serve il pretraining
DA ZERO (costo frontier, non coerente alla tua scala).

Requisiti: GPU grande (es. A100/H100 80GB) o multi-GPU (accelerate/FSDP). bf16.
Il full FT di 4B richiede ~50-80GB (pesi+grad+stati Adam) -> NON sta in 16GB.

Uso (server):
    accelerate launch training/train_fullft.py --base models/Qwen3-4B \
        --data training/data --out models/FractalNova-4B --epochs 2 --lr 1e-5
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def build_dataset(data_dir, out_jsonl="training/data/_fullft.jsonl"):
    """Unisce tutti i .jsonl (libri + identita' + esempi) in formato chat 'messages'."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        if os.path.basename(fp).startswith("_"):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in o:
                    rows.append({"messages": o["messages"]})
                else:
                    instr = (o.get("instruction") or "").strip()
                    inp = (o.get("input") or "").strip()
                    out = (o.get("output") or "").strip()
                    if instr and out:
                        user = instr if not inp else f"{instr}\n\n{inp}"
                        rows.append({"messages": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": out},
                        ]})
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_jsonl, len(rows)


def rebrand(out_dir):
    """Etichetta cosmetica del modello come FractalNova (model_type resta qwen3, necessario)."""
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_name_or_path"] = "FractalNova-4B"
        cfg["model_name"] = "FractalNova-4B"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "FRACTALNOVA.txt"), "w", encoding="utf-8") as f:
        f.write("FractalNova-4B — pesi full fine-tuned. Identita': FractalNova.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen3-4B")
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="models/FractalNova-4B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    args = ap.parse_args()

    data_file, n = build_dataset(args.data)
    print(f"[fullft] dataset: {n} esempi -> {data_file}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    def fmt(batch):
        return [tok.apply_chat_template(m, tokenize=False) for m in batch["messages"]]

    sft = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_seq_length=args.max_seq,
        packing=True,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=sft,
        train_dataset=load_dataset("json", data_files=data_file, split="train"),
        processing_class=tok, formatting_func=fmt,   # NIENTE peft_config -> FULL fine-tuning
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    rebrand(args.out)
    print(f"[fullft] FractalNova-4B (pesi NUOVI, full fine-tuned) -> {args.out}")


if __name__ == "__main__":
    main()
