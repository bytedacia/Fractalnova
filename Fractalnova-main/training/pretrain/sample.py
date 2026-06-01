"""
FractalNova-Core · generazione di testo da un checkpoint addestrato da zero.

Uso:
    python training/pretrain/sample.py \
        --ckpt training/pretrain/outputs/core-124m/ckpt.pt \
        --tokenizer training/pretrain/artifacts/tokenizer.json \
        --prompt "C'era una volta" --max-new-tokens 200
"""
import argparse

import torch
from tokenizers import Tokenizer

from model import FractalNovaCore, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model = FractalNovaCore(GPTConfig(**ckpt["model_args"]))
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(device)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ids = tokenizer.encode(args.prompt).ids if args.prompt else [ckpt.get("eos_id", 0) or 0]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
