"""
FractalNova-Pro · fine-tuning QLoRA (4-bit) di un modello open multilingua.

Specializza Qwen3-4B (multilingua) nella scrittura/editing di libri multi-task
e multilingua. Tarato per stare in 16 GB.

Uso:
    python training/train_qlora.py --config training/configs/qlora_5060ti.yaml

Prima crea i dati:
    python training/prepare_dataset.py --inputs training/data/sample_books_it.jsonl \
        training/data/sample_books_multi.jsonl --out-dir training/data
"""
import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_formatting_func(tokenizer):
    """Rende ogni esempio (campo 'messages') come stringa col chat template del modello."""
    def fmt(batch):
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
    return fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/qlora_5060ti.yaml")
    ap.add_argument("--resume", action="store_true", help="riprendi dall'ultimo checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        print("[ATTENZIONE] CUDA non disponibile: il QLoRA 4-bit richiede una GPU NVIDIA.")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Modello base in 4-bit (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=DTYPE[cfg.get("bnb_4bit_compute_dtype", "bfloat16")],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model.config.use_cache = False  # incompatibile con gradient checkpointing

    # --- LoRA ---
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Dati ---
    data_files = {"train": cfg["train_file"]}
    eval_file = cfg.get("eval_file")
    has_eval = bool(eval_file) and os.path.exists(eval_file)
    if has_eval:
        data_files["validation"] = eval_file
    ds = load_dataset("json", data_files=data_files)

    # --- Config SFT ---
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.get("eval_steps", 100) if has_eval else None,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.get("optim", "paged_adamw_8bit"),
        bf16=cfg.get("bf16", True),
        max_seq_length=cfg.get("max_seq_len", 1024),
        packing=cfg.get("packing", False),
        seed=cfg.get("seed", 42),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_eval else None,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=build_formatting_func(tokenizer),
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"[ok] adapter LoRA salvato in {cfg['output_dir']}")
    print("     Per il deploy: training/merge_and_export.py  |  Per provarlo: training/infer.py")


if __name__ == "__main__":
    main()
