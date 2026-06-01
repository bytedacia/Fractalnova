"""
FractalNova · generatore di libro coerente con un modello causale locale.

Usa un modello (es. models/Qwen3-4B) in bf16 sulla GPU (4B stanno in 16GB, niente
bitsandbytes). Scrive capitolo per capitolo mantenendo continuita' di trama, con
contenuti adatti a un pubblico generale (safety nel system prompt).

Esempi:
    python inference/write_book.py --smoke
    python inference/write_book.py --title "Il Faro di Marta" --genre narrativa --chapters 20 --words-per-chapter 900
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "Sei FractalNova, autore professionista. Scrivi narrativa coerente, ricca e naturale "
    "in italiano, con voce e ritmo curati. Mantieni continuita' di trama, personaggi e tono "
    "tra i capitoli. I contenuti devono essere adatti a un pubblico generale: niente materiale "
    "esplicito, illegale o dannoso."
)


def load(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    try:  # transformers >= 5 usa `dtype`; versioni precedenti `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def _inputs(tok, messages, device):
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(device)


def gen(model, tok, user: str, max_new_tokens: int = 1200, temperature: float = 0.9, top_p: float = 0.92) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    enc = _inputs(tok, messages, model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-4B")
    ap.add_argument("--title", default="Il Faro di Marta")
    ap.add_argument("--genre", default="narrativa")
    ap.add_argument("--chapters", type=int, default=8)
    ap.add_argument("--words-per-chapter", type=int, default=800)
    ap.add_argument("--out", default="generated_books/libro.md")
    ap.add_argument("--smoke", action="store_true", help="solo un breve test di coerenza")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load(args.model)
    print(f"[load] modello caricato in {time.time() - t0:.0f}s su {model.device}")

    if args.smoke:
        txt = gen(model, tok,
                  f"Scrivi l'incipit (circa 250 parole) di un romanzo intitolato '{args.title}', "
                  f"genere {args.genre}. Solo il testo.", max_new_tokens=400)
        print("\n===== SMOKE TEST =====\n" + txt + "\n======================")
        return

    # 1) scaletta
    outline = gen(model, tok,
                  f"Crea la scaletta di {args.chapters} capitoli per un romanzo '{args.title}' "
                  f"({args.genre}). Elenca SOLO i titoli, uno per riga.",
                  max_new_tokens=700, temperature=0.7)
    titles = [ln.strip(" -*0123456789.").strip() for ln in outline.splitlines() if ln.strip()][:args.chapters]
    if not titles:
        titles = [f"Capitolo {i}" for i in range(1, args.chapters + 1)]

    story = f"# {args.title}\n\n"
    prev = ""
    for i, t in enumerate(titles, 1):
        prev_summary = prev[:800] if prev else "(inizio del libro)"
        ch = gen(model, tok,
                 f"Romanzo '{args.title}' ({args.genre}). Scrivi il Capitolo {i}: «{t}», "
                 f"circa {args.words_per_chapter} parole, coerente con quanto precede.\n\n"
                 f"Sintesi del capitolo precedente:\n{prev_summary}\n\n"
                 f"Scrivi SOLO il testo del capitolo.",
                 max_new_tokens=int(args.words_per_chapter * 2.2), temperature=0.9)
        story += f"\n## {t}\n\n{ch}\n"
        prev = ch
        print(f"[cap {i}/{len(titles)}] {t} — {len(ch.split())} parole")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(story)
    words = len(story.split())
    print(f"\n[fatto] ~{words} parole (~{max(1, words // 300)} pagine) -> {args.out}")


if __name__ == "__main__":
    main()
