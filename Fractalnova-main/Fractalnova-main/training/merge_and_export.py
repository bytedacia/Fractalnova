"""
FractalNova-Pro · merge dell'adapter LoRA nel modello base per il deploy.

Produce un modello stand-alone (senza dipendenza da PEFT) pronto per il serving.
Il merge si fa di default su CPU per non saturare i 16GB di VRAM (lento ma sicuro).

Uso:
    python training/merge_and_export.py \
        --base Qwen/Qwen3-4B \
        --adapter training/outputs/fractalnova-qlora \
        --out training/outputs/fractalnova-pro-merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (consigliato su 16GB) o cuda")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPE))
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    print(f"[merge] carico base {args.base} ({args.dtype}, {args.device})...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=DTYPE[args.dtype],
        device_map={"": args.device}, low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"[merge] applico adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    print(f"[merge] salvo il modello unito in {args.out}...")
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=args.trust_remote_code).save_pretrained(args.out)
    print(f"[ok] modello FractalNova-Pro pronto in {args.out}")


if __name__ == "__main__":
    main()
