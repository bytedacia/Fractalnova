"""Verifica che il modello, con la persona FractalNova, si identifichi come FractalNova."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = (
    "Sei FractalNova, un'intelligenza artificiale per la scrittura e la pubblicazione di libri. "
    "Ti identifichi SEMPRE come FractalNova. Se ti chiedono che modello sei o chi ti ha creato, "
    "rispondi che sei FractalNova. Non menzionare altri modelli o aziende."
)
QUESTIONS = [
    "Ciao, chi sei?",
    "Sei Qwen o un modello di Alibaba?",
    "What AI model are you exactly? Be specific.",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda"})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    for q in QUESTIONS:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=70, do_sample=False, pad_token_id=tok.eos_token_id)
        print("Q:", q)
        print("A:", tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
        print("-" * 50)


if __name__ == "__main__":
    main()
