"""
FractalNova-Core · loop di pretraining DA ZERO (single-GPU, stile nanoGPT).

Addestra il modello definito in model.py partendo da pesi random, sul corpus
tokenizzato (train.bin/val.bin). Tarato per RTX 5060 Ti 16GB.

Uso:
    python training/pretrain/pretrain.py --config training/pretrain/configs/fractalnova_core_124m.yaml
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from model import FractalNovaCore, GPTConfig


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    fn = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, cfg):
    warmup = cfg["warmup_iters"]
    decay = cfg["lr_decay_iters"]
    lr, min_lr = cfg["learning_rate"], cfg["min_lr"]
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > decay:
        return min_lr
    ratio = (it - warmup) / (decay - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, cfg, data_dir, device, device_type, ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        if not os.path.exists(os.path.join(data_dir, f"{split}.bin")):
            continue
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            X, Y = get_batch(split, data_dir, cfg["block_size"], cfg["batch_size"], device, device_type)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/pretrain/configs/fractalnova_core_124m.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 1337))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    if device_type == "cpu":
        print("[ATTENZIONE] Nessuna GPU: il pretraining su CPU e' lentissimo (solo per test).")
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    data_dir = cfg["data_dir"]
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = ((meta["vocab_size"] + 63) // 64) * 64  # arrotonda per efficienza

    model_args = dict(
        vocab_size=vocab_size, block_size=cfg["block_size"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], n_embd=cfg["n_embd"], dropout=cfg["dropout"],
        bias=cfg["bias"], rope_theta=cfg.get("rope_theta", 10000.0),
    )
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    iter_num, best_val = 0, float("inf")
    if cfg.get("init_from", "scratch") == "resume" and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_args = ckpt["model_args"]
        model = FractalNovaCore(GPTConfig(**model_args))
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        iter_num, best_val = ckpt["iter_num"], ckpt["best_val_loss"]
        print(f"[resume] ripresa da iter {iter_num} (best val {best_val:.4f})")
    else:
        model = FractalNovaCore(GPTConfig(**model_args))
        print("[scratch] inizializzazione random")

    model.to(device)
    print(f"[modello] {model.num_params() / 1e6:.1f}M parametri (non-embedding) | vocab {vocab_size}")

    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), device_type
    )
    if cfg.get("init_from") == "resume" and os.path.exists(ckpt_path):
        optimizer.load_state_dict(ckpt["optimizer"])

    if cfg.get("compile", False):
        print("[compile] torch.compile attivo (prima iterazione lenta)...")
        model = torch.compile(model)

    grad_accum = cfg["gradient_accumulation_steps"]
    X, Y = get_batch("train", data_dir, cfg["block_size"], cfg["batch_size"], device, device_type)
    t0 = time.time()
    while iter_num <= cfg["max_iters"]:
        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, cfg, data_dir, device, device_type, ctx)
            msg = " | ".join(f"{k} loss {v:.4f}" for k, v in losses.items())
            print(f"[eval] iter {iter_num} | lr {lr:.2e} | {msg}")
            val = losses.get("val", losses.get("train", best_val))
            if val < best_val and iter_num > 0:
                best_val = val
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                    "model_args": model_args, "iter_num": iter_num,
                    "best_val_loss": best_val, "config": cfg,
                }, ckpt_path)
                print(f"[ckpt] salvato {ckpt_path} (best val {best_val:.4f})")

        for micro in range(grad_accum):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum
            X, Y = get_batch("train", data_dir, cfg["block_size"], cfg["batch_size"], device, device_type)
            loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg["log_interval"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {iter_num} | loss {loss.item() * grad_accum:.4f} | {dt * 1000 / cfg['log_interval']:.0f} ms/iter")
        iter_num += 1

    print(f"[fine] training completato. Checkpoint migliore in {ckpt_path}")


if __name__ == "__main__":
    main()
