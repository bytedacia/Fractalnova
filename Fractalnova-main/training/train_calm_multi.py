"""
FractalNova · addestramento dei BRIDGE del CALM CONDIVISO (3 modelli).

Anchor + augmenting CONGELATI; si addestrano SOLO i bridge cross-attention
(gli unici PESI NUOVI degli agenti). Output: <out>/bridges.pt + calm_config.json.

Server (~30GB VRAM). Esempio:
    python training/train_calm_multi.py \
        --anchor models/Mistral-7B-Instruct-v0.3 \
        --aug models/Nemotron-Mini-4B-Instruct models/Phi-3.5-mini-instruct \
        --data agents/data --epochs 2 --max-seq 512 \
        --out agents/calm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from inference.calm_multi import MultiCALM  # noqa: E402


def _rows(data_path: str):
    files = [data_path] if os.path.isfile(data_path) else sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _to_messages(row):
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    if {"system", "user", "assistant"} & set(row):
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        if row.get("user"):
            msgs.append({"role": "user", "content": row["user"]})
        if row.get("assistant"):
            msgs.append({"role": "assistant", "content": row["assistant"]})
        return msgs or None
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if not instr or not out:
        return None
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="models/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--aug", nargs="+", default=["models/Nemotron-Mini-4B-Instruct", "models/Phi-3.5-mini-instruct"])
    ap.add_argument("--data", default="agents/data")
    ap.add_argument("--out", default="agents/calm")
    ap.add_argument("--connect-every", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    model = MultiCALM(args.anchor, args.aug, connect_every=args.connect_every)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"[calm-multi] {len(args.aug)} augmenting · bridge: {n / 1e6:.1f}M parametri addestrabili "
          f"su layer {model.connect_layers}")

    examples = []
    for row in _rows(args.data):
        msgs = _to_messages(row)
        if not msgs:
            continue
        text = model.anchor_tok.apply_chat_template(msgs, tokenize=False)
        a = model.anchor_tok(text, truncation=True, max_length=args.max_seq, return_tensors="pt").input_ids[0]
        if a.numel() >= 8:
            examples.append((a, text))
    if not examples:
        raise SystemExit(f"Nessun esempio valido in {args.data}")
    print(f"[calm-multi] {len(examples)} esempi di training")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    dev = model.device
    autocast = torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16)
    n_aug = len(model.augs)

    for ep in range(args.epochs):
        random.shuffle(examples)
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for i, (a, text) in enumerate(examples):
            a_ids = a.unsqueeze(0).to(dev)
            with autocast:
                out = model(a_ids, aug_texts=[text] * n_aug, labels=a_ids)
                loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if (i + 1) % 20 == 0:
                print(f"  ep{ep + 1} {i + 1}/{len(examples)} loss {running / 20:.4f}")
                running = 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.bridges.state_dict(), os.path.join(args.out, "bridges.pt"))
    with open(os.path.join(args.out, "calm_config.json"), "w", encoding="utf-8") as f:
        json.dump({"anchor": args.anchor, "aug": args.aug,
                   "connect_every": args.connect_every, "connect_layers": model.connect_layers}, f, indent=2)
    print(f"[calm-multi] bridge salvati -> {args.out}/bridges.pt")


if __name__ == "__main__":
    main()
