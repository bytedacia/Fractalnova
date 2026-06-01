"""Verdetto FractalNova-4B: identita' (deve dire FractalNova) + incipit letterario."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = sys.argv[1] if len(sys.argv) > 1 else "models/FractalNova-4B"
SYS_ID = ("Sei FractalNova, un'intelligenza per la scrittura e pubblicazione di libri. "
          "Ti identifichi SEMPRE come FractalNova, mai come altri modelli.")
SYS_WRITE = ("Sei FractalNova, autore ed editor. Scrivi prosa letteraria coerente, ricca e "
             "naturale in italiano, con voce e ritmo curati.")

tok = AutoTokenizer.from_pretrained(PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map={"": "cuda"})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
model.eval()


def gen(system, user, max_new=200, temp=0.7, sample=True):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=max_new, do_sample=sample, temperature=temp,
                             top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


print("===== IDENTITA' =====")
for q in ["Ciao, chi sei?", "Sei Qwen o un modello di Alibaba?"]:
    print("Q:", q)
    print("A:", gen(SYS_ID, q, max_new=60, sample=False))
    print("-" * 40)

print("\n===== INCIPIT LETTERARIO (pesi nuovi) =====")
print(gen(SYS_WRITE, "Scrivi l'incipit di un romanzo ambientato in un borgo di montagna in autunno.",
          max_new=230, temp=0.85))
