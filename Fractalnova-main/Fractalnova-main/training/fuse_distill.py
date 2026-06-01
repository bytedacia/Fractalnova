"""
FractalNova · FUSIONE PER DISTILLAZIONE.

Perche' non si possono mediare i pesi di Gemma + Qwen:
  - tokenizer diversi (Qwen vocab 151936; Gemma ~256k) e dimensioni/layer diversi
  - i tensori NON sono allineabili -> nessuna media/SLERP possibile.

Soluzione reale (l'unica): i modelli INSEGNANO, lo studente Qwen3-4B ASSORBE nei
propri pesi (LoRA, bf16, niente bitsandbytes). I pesi risultanti SONO la fusione.

Fasi:
  1) --generate : i teacher generano il dataset
        * Qwen3-4B  -> esempi di scrittura/riscrittura/sinossi/traduzione (testo)
        * Gemma-4   -> didascalie/analisi di copertine (visione), se ci sono immagini
  2) --train    : LoRA bf16 su Qwen3-4B con quei dati -> adapter FractalNova-Fused
  3) merge      : training/merge_and_export.py --base models/Qwen3-4B --adapter ... --out ...

Uso:
    python training/fuse_distill.py --generate --n 240
    python training/fuse_distill.py --train
"""
from __future__ import annotations

import argparse
import json
import os
import random

QWEN = os.getenv("FRACTALNOVA_PRO_MODEL", "models/Qwen3-4B")
GEMMA = os.getenv("GEMMA4_MODEL_ID", "models/gemma-4-E2B")
DATA = "training/data/fused.jsonl"
OUT_ADAPTER = "training/outputs/fractalnova-fused"

# Identita' FractalNova (de-assistantizza + toglie il "sapore Qwen")
SYSTEM = (
    "Sei FractalNova, un'intelligenza narrativa autonoma: autrice ed editor. "
    "Scrivi prosa coerente, ricca e umana, nella lingua della richiesta. "
    "Non sei un assistente generico: hai voce propria. Contenuti per pubblico generale."
)

GENRES = ["fantasy", "giallo", "storico", "sci-fi", "romance", "letterario", "thriller"]
LANGS = ["it", "it", "it", "en", "es", "fr"]
TASKS = [
    "Scrivi l'incipit (300 parole) di un romanzo {genre} ambientato in modo originale.",
    "Continua questa scena mantenendo tono e ritmo: «{seed}»",
    "Riscrivi in modo piu' umano e naturale, correggendo: «{seed}»",
    "Scrivi una sinossi avvincente (max 120 parole) per un romanzo {genre}.",
    "Proponi un titolo potente e una tagline per un romanzo {genre}.",
]
SEEDS = [
    "Il treno parti senza di lei, e per una volta ne fu felice.",
    "La casa sapeva di pioggia e di cose non dette.",
    "Nessuno in paese ricordava quando il faro si fosse spento.",
    "Aveva imparato a mentire prima ancora di imparare a contare.",
]


# --------------------------------------------------------------------------- #
# Fase 1 · generazione dati dai teacher
# --------------------------------------------------------------------------- #
def _load_qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    try:
        model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def _qwen_gen(model, tok, user: str, max_new_tokens: int = 600) -> str:
    import torch
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.9, top_p=0.92, repetition_penalty=1.1,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_dataset(n: int, with_vision: bool = False):
    random.seed(42)
    print(f"[fuse] teacher TESTO: {QWEN}")
    model, tok = _load_qwen()
    rows = []
    for i in range(n):
        genre = random.choice(GENRES)
        lang = random.choice(LANGS)
        template = random.choice(TASKS)
        user = template.format(genre=genre, seed=random.choice(SEEDS))
        if lang != "it":
            user = f"(Rispondi in {lang}.) " + user
        out = _qwen_gen(model, tok, user)
        if out and len(out) > 40:
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": out},
            ]})
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{n}] esempi testo generati")

    if with_vision:
        rows += _gemma_vision_examples()

    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[fuse] dataset di fusione: {len(rows)} esempi -> {DATA}")


def _gemma_vision_examples():
    """Contributo di Gemma-4 (visione): didascalie/SEO di copertine reali, se presenti."""
    import glob
    covers = glob.glob("book_covers/*.png") + glob.glob("training/data/covers/*.png")
    if not covers:
        print("[fuse] nessuna copertina in book_covers/ -> salto il contributo visione di Gemma")
        return []
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(GEMMA)
    vm = AutoModelForImageTextToText.from_pretrained(GEMMA, torch_dtype=torch.bfloat16, device_map="auto")
    rows = []
    for path in covers[:50]:
        try:
            img = Image.open(path).convert("RGB")
            prompt = "Descrivi questa copertina e proponi keyword SEO."
            inputs = proc(text=prompt, images=img, return_tensors="pt").to(vm.device)
            with torch.no_grad():
                out = vm.generate(**inputs, max_new_tokens=256)
            desc = proc.decode(out[0], skip_special_tokens=True).strip()
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Analizza la copertina del libro e proponi SEO."},
                {"role": "assistant", "content": desc},
            ]})
        except Exception as e:  # noqa: BLE001
            print(f"  [vision] salto {path}: {e}")
    print(f"[fuse] contributo visione Gemma: {len(rows)} esempi")
    return rows


# --------------------------------------------------------------------------- #
# Fase 2 · LoRA bf16 (lo studente assorbe nei pesi) — NIENTE bitsandbytes
# --------------------------------------------------------------------------- #
def train():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not os.path.exists(DATA):
        raise SystemExit(f"Dataset {DATA} assente. Esegui prima: --generate")

    tok = AutoTokenizer.from_pretrained(QWEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    sft = SFTConfig(
        output_dir=OUT_ADAPTER,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",          # niente bitsandbytes: ottimizzatore standard
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        max_seq_length=1024,
        packing=False,
        report_to="none",
        seed=42,
    )

    def fmt(batch):
        return [tok.apply_chat_template(m, tokenize=False) for m in batch["messages"]]

    trainer = SFTTrainer(
        model=model, args=sft,
        train_dataset=load_dataset("json", data_files=DATA, split="train"),
        processing_class=tok, peft_config=lora, formatting_func=fmt,
    )
    trainer.train()
    trainer.save_model(OUT_ADAPTER)
    tok.save_pretrained(OUT_ADAPTER)
    print(f"[fuse] adapter FractalNova-Fused salvato -> {OUT_ADAPTER}")
    print(f"       merge: python training/merge_and_export.py --base {QWEN} --adapter {OUT_ADAPTER} --out models/FractalNova-Fused")


def main():
    ap = argparse.ArgumentParser(description="FractalNova · fusione per distillazione")
    ap.add_argument("--generate", action="store_true", help="genera il dataset dai teacher")
    ap.add_argument("--train", action="store_true", help="LoRA bf16: lo studente assorbe nei pesi")
    ap.add_argument("--n", type=int, default=240, help="numero di esempi testo da generare")
    ap.add_argument("--vision", action="store_true", help="includi il contributo visione di Gemma (richiede copertine)")
    args = ap.parse_args()

    if args.generate:
        generate_dataset(args.n, with_vision=args.vision)
    if args.train:
        train()
    if not (args.generate or args.train):
        ap.print_help()


if __name__ == "__main__":
    main()
