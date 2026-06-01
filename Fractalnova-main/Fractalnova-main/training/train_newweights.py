"""
FractalNova · PESI NUOVI su 16GB (LoRA bf16 -> MERGE).

Il full fine-tune di 4B non sta in 16GB. Ma una LoRA bf16 + MERGE produce comunque
PESI NUOVI: i delta vengono fusi nelle matrici (q,k,v,o,gate,up,down di tutti i layer),
quindi i pesi risultanti NON sono identici alla base, con identita'/voce FractalNova.
(Per pesi ancora piu' diversi -> training/train_fullft.py sul server.)

Uso:
    python training/train_newweights.py --base models/Qwen3-4B --data training/data --out models/FractalNova-4B
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_examples(tok, data_dir, maxlen):
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
                    msgs = o["messages"]
                else:
                    instr = (o.get("instruction") or "").strip()
                    inp = (o.get("input") or "").strip()
                    out = (o.get("output") or "").strip()
                    if not out:
                        continue
                    user = instr if not inp else f"{instr}\n\n{inp}"
                    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
                text = tok.apply_chat_template(msgs, tokenize=False)
                ids = tok(text, truncation=True, max_length=maxlen, return_tensors="pt").input_ids[0]
                if ids.numel() >= 8:
                    rows.append(ids)
    return rows


def rebrand(out_dir):
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_name_or_path"] = "FractalNova-4B"
        cfg["model_name"] = "FractalNova-4B"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen3-4B")
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="models/FractalNova-4B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    examples = load_examples(tok, args.data, args.maxlen)
    if not examples:
        raise SystemExit(f"Nessun esempio in {args.data}")
    print(f"[newweights] {len(examples)} esempi di training (maxlen {args.maxlen})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        random.shuffle(examples)
        opt.zero_grad(set_to_none=True)
        run = 0.0
        for i, ids in enumerate(examples):
            x = ids.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if (i + 1) % 20 == 0:
                print(f"  ep{ep+1} {i+1}/{len(examples)} loss {run/20:.4f}")
                run = 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)

    # MERGE: fonde i delta LoRA nei pesi -> pesi NUOVI
    print("[newweights] merge dei delta nei pesi...")
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merged = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    rebrand(args.out)
    print(f"[newweights] FractalNova-4B (PESI NUOVI) salvato -> {args.out}")


if __name__ == "__main__":
    main()
