"""
FractalNova-Pro · valutazione del modello fine-tunato.

Due metriche:
  1) Perplexity sul set di validazione (val.jsonl) -- metrica quantitativa relativa.
  2) Generazioni qualitative multilingua su prompt fissi -- per ispezione umana.

Esempio:
    python training/evaluate.py --base Qwen/Qwen3-4B \
        --adapter training/outputs/fractalnova-qlora --load-4bit \
        --eval-file training/data/val.jsonl --max-samples 100
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SYSTEM_PROMPT, load_model  # noqa: E402

QUALITATIVE_PROMPTS = [
    ("it", "Scrivi l'incipit di un romanzo storico ambientato a Venezia nel 1600."),
    ("en", "Continue, in a literary tone: 'The train left without her, and for once she was glad.'"),
    ("es", "Reescribe de forma mas humana: 'El protagonista estaba triste y no sabia que hacer.'"),
    ("fr", "Redige une synopsis de 60 mots pour un thriller psychologique."),
    ("de", "Schlage einen Buchtitel und eine Tagline fuer einen Fantasyroman vor."),
]


@torch.no_grad()
def perplexity(model, tokenizer, examples, max_length):
    losses = []
    for ex in examples:
        rendered = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        if enc.input_ids.size(1) < 2:
            continue
        out = model(**enc, labels=enc.input_ids)
        losses.append(out.loss.item())
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--eval-file", default="training/data/val.jsonl")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load_model(
        model_path=args.model, base=args.base, adapter=args.adapter,
        load_4bit=args.load_4bit, trust_remote_code=args.trust_remote_code,
    )

    # 1) Perplexity
    if os.path.exists(args.eval_file):
        examples = []
        with open(args.eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        examples = examples[: args.max_samples]
        ppl = perplexity(model, tokenizer, examples, args.max_length)
        print(f"\n=== Perplexity su {len(examples)} esempi: {ppl:.2f} (piu' basso = meglio) ===\n")
    else:
        print(f"[skip] {args.eval_file} non trovato: salto la perplexity.")

    # 2) Generazioni qualitative multilingua
    print("=== Generazioni qualitative (multilingua) ===")
    for lang, prompt in QUALITATIVE_PROMPTS:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=160, do_sample=True, temperature=0.8,
                                 top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
        print(f"\n[{lang}] {prompt}\n -> {text}")


if __name__ == "__main__":
    main()
