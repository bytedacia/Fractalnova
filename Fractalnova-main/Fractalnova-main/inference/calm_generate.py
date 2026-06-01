"""
FractalNova-CALM · genera testo dal modello composto (Qwen3-4B + augmenting + bridge addestrato).

Uso:
    python inference/calm_generate.py --dir training/outputs/fractalnova-calm \
        --prompt "Scrivi l'incipit di un romanzo fantasy." --max-new-tokens 250
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.calm import FractalNovaCALM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="training/outputs/fractalnova-calm")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "calm_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = FractalNovaCALM(cfg["anchor"], cfg["aug"], connect_every=cfg["connect_every"])
    state = torch.load(os.path.join(args.dir, "bridge.pt"), map_location=model.device)
    model.bridges.load_state_dict(state)
    print(f"[calm-gen] FractalNova-CALM pronto (anchor={cfg['anchor']}, aug={cfg['aug']})\n")

    out = model.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print("===== FractalNova-CALM =====")
    print(out)
    print("============================")


if __name__ == "__main__":
    main()
