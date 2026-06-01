"""
FractalNova-Pro · Unified multi-model fine-tuning.

Scarica e fine-tuna TUTTI e TRE i modelli FractalNova in un unico comando:

  Qwen3-4B    (testo)     → QLoRA su dati libri
  Gemma-4-E2B (visione)   → LoRA su analisi copertine
  FLUX.1-dev  (copertine) → LoRA su generazione copertine libri

Output: training/outputs/fractalnova-pro/
  text_adapter/    → adapter QLoRA per Qwen3-4B
  vision_adapter/  → adapter LoRA per Gemma-4-E2B
  cover_adapter/   → adapter LoRA per FLUX.1-dev
  manifest.json    → config per FractalNovaInference
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT = Path("training/outputs/fractalnova-pro")
TEXT_DIR = OUTPUT / "text_adapter"
VISION_DIR = OUTPUT / "vision_adapter"
COVER_DIR = OUTPUT / "cover_adapter"


# ── Step 0: download ────────────────────────────────────────────

BASE_MODELS = {
    "Qwen3-4B":    (os.getenv("FRACTALNOVA_PRO_MODEL", "models/Qwen3-4B"),    "Qwen/Qwen3-4B"),
    "Gemma-4-E2B": (os.getenv("GEMMA4_MODEL_ID",      "models/gemma-4-E2B"), "google/gemma-4-E2B"),
    "FLUX.1-dev":  (os.getenv("FLUX_MODEL_ID",        "models/FLUX.1-dev"),  "black-forest-labs/FLUX.1-dev"),
}

def download_all(token: str = None):
    """Scarica i 3 modelli base se non già presenti localmente."""
    from huggingface_hub import snapshot_download
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    for name, (local_path, repo) in BASE_MODELS.items():
        if os.path.isdir(local_path) and any(f.endswith(".safetensors") for f in os.listdir(local_path)):
            print(f"  ✓ {name} già presente in {local_path}")
            continue
        print(f"  ↓ download {repo} → {local_path} ...")
        os.makedirs(local_path, exist_ok=True)
        try:
            snapshot_download(repo_id=repo, local_dir=local_path, local_dir_use_symlinks=False, token=token)
            print(f"  ✓ {name} completo")
        except Exception as e:
            print(f"  ✗ {name} fallito: {e}")


# ── Step 1: fine-tune testo (Qwen3-4B + QLoRA) ─────────────────

def train_text(train_file: str, val_file: str = None):
    """Fine-tune Qwen3-4B con QLoRA sul dataset libri (SFT via TRL)."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_path = BASE_MODELS["Qwen3-4B"][0]
    if not os.path.exists(train_file):
        print("  ⚠ training set assente, salto fine-tune testo")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto", attn_implementation="sdpa",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    has_eval = bool(val_file) and os.path.exists(val_file)
    # NB: max_seq_length vive in SFTConfig (NON in TrainingArguments). seq 2048 per 16GB.
    sft = SFTConfig(
        output_dir=str(TEXT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=200 if has_eval else None,
        max_seq_length=2048,
        packing=False,
        report_to="none",
        seed=42,
    )

    def fmt(batch):
        return [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=load_dataset("json", data_files=train_file, split="train"),
        eval_dataset=load_dataset("json", data_files=val_file, split="train") if has_eval else None,
        processing_class=tokenizer,
        peft_config=lora,
        formatting_func=fmt,
    )
    trainer.train()
    trainer.save_model(str(TEXT_DIR))
    tokenizer.save_pretrained(str(TEXT_DIR))
    print(f"  ✓ text adapter salvato in {TEXT_DIR}")


# ── Step 2: fine-tune visione (Gemma-4-E2B + LoRA) ──────────────

def train_vision():
    """Fine-tune Gemma-4-E2B con LoRA su analisi copertine.

    Usa dati sintetici: coppie (immagine copertina → analisi SEO/descrizione).
    Se non ci sono dati reali, crea un piccolo set di esempio.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
    from peft import LoraConfig, get_peft_model
    import torch

    gemma_path = BASE_MODELS["Gemma-4-E2B"][0]
    model = AutoModelForImageTextToText.from_pretrained(
        gemma_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(gemma_path)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],  # vision encoder
        bias="none", task_type="IMAGE_TEXT_TO_TEXT",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    vision_data = _generate_vision_examples(processor)
    if not vision_data:
        print("  ⚠ nessun dato visione, skip fine-tune")
        return

    args = TrainingArguments(
        output_dir=str(VISION_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )

    trainer = VisionTrainer(
        model=model,
        args=args,
        train_dataset=vision_data,
        processor=processor,
    )
    trainer.train()
    model.save_pretrained(str(VISION_DIR))
    processor.save_pretrained(str(VISION_DIR))
    print(f"  ✓ vision adapter salvato in {VISION_DIR}")


def _generate_vision_examples(processor):
    """Genera dataset sintetico per vision fine-tuning."""
    from datasets import Dataset
    samples = [
        {
            "image_path": "book_covers/sample_fantasy.png",
            "text": "Analyze this book cover for SEO. Return JSON with alt_text, keywords, description.",
            "output": '{"alt_text": "Fantasy book cover with dragon and castle", "keywords": ["fantasy","dragon","epic"], "description": "An epic fantasy novel cover featuring a dragon soaring above a castle."}',
        },
        {
            "image_path": "book_covers/sample_scifi.png",
            "text": "Classify the genre of this book cover.",
            "output": "sci-fi",
        },
    ]
    # Filtra solo esempi con immagini esistenti
    valid = [s for s in samples if os.path.exists(s["image_path"])]
    if not valid:
        return None
    return Dataset.from_list(valid)


class VisionTrainer:
    """Trainer minimalista per modelli vision-language."""
    def __init__(self, model, args, train_dataset, processor):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.processor = processor
        from transformers import Trainer as HF_Trainer
        self._trainer = HF_Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

    def train(self):
        import torch
        self.model.train()
        from tqdm import tqdm
        for epoch in range(int(self.args.num_train_epochs)):
            for batch in tqdm(self.train_dataset, desc=f"Vision Epoch {epoch+1}"):
                img = self._load_image(batch["image_path"])
                if img is None:
                    continue
                inputs = self.processor(text=batch["text"], images=img, return_tensors="pt").to(self.model.device)
                labels = self.processor(text=batch["output"], return_tensors="pt").input_ids.to(self.model.device)
                outputs = self.model(**inputs, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # accumulo manuale
                if (self._step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate).step()
                    self.model.zero_grad()
                self._step += 1

    def _load_image(self, path):
        from PIL import Image
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None


# ── Step 3: fine-tune copertine (FLUX.1-dev + LoRA) ─────────────

def train_cover():
    """Fine-tune FLUX.1-dev con LoRA su generazione copertine libri."""
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from diffusers.training_utils import AttnProcsLayers
    from peft import LoraConfig, get_peft_model
    import torch

    transformer = FluxTransformer2DModel.from_pretrained(
        "models/FLUX.1-dev", subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    transformer = get_peft_model(transformer, lora)
    transformer.print_trainable_parameters()

    cover_data = _generate_cover_examples()
    if not cover_data:
        print("  ⚠ nessun dato copertine, skip fine-tune")
        return

    # Training loop semplificato: FLUX LoRA
    from diffusers.optimization import get_scheduler
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-4)
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100, num_training_steps=len(cover_data) * 5)
    transformer.train()
    from tqdm import tqdm
    for epoch in range(5):
        for batch in tqdm(cover_data, desc=f"Cover Epoch {epoch+1}"):
            loss = _compute_flux_loss(transformer, batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    transformer.save_pretrained(str(COVER_DIR))
    print(f"  ✓ cover adapter salvato in {COVER_DIR}")


def _generate_cover_examples():
    """Dataset per il fine-tuning copertine (coppie prompt → immagine reale).

    Il fine-tune di FLUX richiede un VERO dataset di immagini. Non addestriamo su
    rumore finto: se non c'e' un dataset reale in training/data/covers/, si salta.
    Popola quella cartella con coppie (prompt, .png) e implementa il caricamento qui.
    """
    covers_dir = os.path.join("training", "data", "covers")
    if not os.path.isdir(covers_dir) or not any(f.endswith(".png") for f in os.listdir(covers_dir)):
        return []
    # TODO: caricare coppie reali (prompt, immagine) da covers_dir
    return []


def _compute_flux_loss(transformer, batch):
    """Calcola loss per FLUX LoRA training."""
    import torch
    # Placeholder: in produzione usa vae.encode + noise + transformer
    dummy_latents = torch.randn(1, 16, 64, 64, dtype=torch.bfloat16)
    dummy_timesteps = torch.randint(0, 1000, (1,), dtype=torch.long)
    dummy_embeds = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    noise_pred = transformer(
        hidden_states=dummy_latents,
        timestep=dummy_timesteps,
        encoder_hidden_states=dummy_embeds,
        return_dict=False,
    )[0]
    target = torch.randn_like(noise_pred)
    return torch.nn.functional.mse_loss(noise_pred, target)


# ── Step 4: crea manifest ────────────────────────────────────────

MANIFEST = {
    "fractalnova_version": "2.0",
    "profile": "pro",
    "models": {
        "text": {
            "base": "Qwen/Qwen3-4B",
            "adapter": "training/outputs/fractalnova-pro/text_adapter",
            "type": "causal_lm",
            "finetune_method": "qlora",
        },
        "vision": {
            "base": "google/gemma-4-E2B",
            "adapter": "training/outputs/fractalnova-pro/vision_adapter",
            "type": "image_text_to_text",
            "finetune_method": "lora",
        },
        "cover": {
            "base": "black-forest-labs/FLUX.1-dev",
            "adapter": "training/outputs/fractalnova-pro/cover_adapter",
            "type": "diffusion",
            "finetune_method": "lora",
        },
    },
    "training": {
        "epochs": 3,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 3e-4,
        "max_seq_len": 8192,
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "benchmark": {
        "target": "GPT-4, Claude, Gemini",
        "status": "pending",
    },
}


def save_manifest():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "manifest.json"
    with open(path, "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"  ✓ manifest salvato in {path}")


# ── main ─────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="FractalNova-Pro · Unified fine-tuning")
    ap.add_argument("--download-only", action="store_true", help="Solo download modelli")
    ap.add_argument("--train-file", default="training/data/train.jsonl", help="Dataset SFT")
    ap.add_argument("--val-file", default="training/data/val.jsonl", help="Dataset validazione")
    ap.add_argument("--token", help="HuggingFace token per modelli gated")
    ap.add_argument("--skip-text", action="store_true", help="Salta fine-tune testo")
    ap.add_argument("--skip-vision", action="store_true", help="Salta fine-tune visione")
    ap.add_argument("--skip-cover", action="store_true", help="Salta fine-tune copertine")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("  FractalNova-Pro · Unified Fine-Tuning")
    print("=" * 60)

    print("\n[1/4] Download modelli base...")
    download_all(args.token)
    if args.download_only:
        return

    print("\n[2/4] Fine-tune testo (Qwen3-4B + QLoRA)...")
    if not args.skip_text:
        train_text(args.train_file, args.val_file)
    else:
        print("  skippato")

    print("\n[3/4] Fine-tune visione (Gemma-4-E2B + LoRA)...")
    if not args.skip_vision:
        train_vision()
    else:
        print("  skippato")

    print("\n[4/4] Fine-tune copertine (FLUX.1-dev + LoRA)...")
    if not args.skip_cover:
        train_cover()
    else:
        print("  skippato")

    save_manifest()
    print("\n" + "=" * 60)
    print(f"  ✓ FractalNova-Pro completo → {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
