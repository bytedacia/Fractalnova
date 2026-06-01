"""
FractalNova · Automated hyperparameter optimization for QLoRA fine-tuning.

Uses Optuna to search for optimal LoRA rank, learning rate, alpha, dropout, and
batch size. Trains on a small subset (10-20%) of data for rapid iteration.

Usage:
    python training/hyperparameter_tuning.py --n-trials 50
    python training/hyperparameter_tuning.py --study-name fractalnova-v1 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import optuna
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

SEARCH_SPACE = {
    "lora_r": {"type": "int", "low": 4, "high": 64, "step": 4},
    "lora_alpha": {"type": "int", "low": 8, "high": 64, "step": 8},
    "lora_dropout": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
    "learning_rate": {"type": "float", "low": 5e-6, "high": 5e-4, "log": True},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
    "warmup_ratio": {"type": "float", "low": 0.01, "high": 0.1},
    "gradient_accumulation_steps": {"type": "int", "low": 2, "high": 32, "step": 2},
    "lora_target_modules_count": {"type": "int", "low": 2, "high": 7, "step": 1},
}

TARGET_MODULE_OPTIONS = [
    ["q_proj", "v_proj"],
    ["q_proj", "k_proj", "v_proj"],
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
]


def objective(trial: optuna.Trial, cfg_base: Dict, train_dataset, val_dataset, tokenizer) -> float:
    lora_r = trial.suggest_int("lora_r", 4, 64, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1)
    grad_accum = trial.suggest_int("gradient_accumulation_steps", 2, 32, step=2)
    module_idx = trial.suggest_int("lora_target_modules_count", 0, len(TARGET_MODULE_OPTIONS) - 1)
    target_modules = TARGET_MODULE_OPTIONS[module_idx]

    print(f"\n[trial {trial.number}] lora_r={lora_r} alpha={lora_alpha} lr={learning_rate:.2e} "
          f"wd={weight_decay} warmup={warmup_ratio} accum={grad_accum} modules={len(target_modules)}")

    try:
        # Load model in 4-bit (fresh each trial to avoid memory leaks)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg_base["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=cfg_base.get("trust_remote_code", False),
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        # Quick training (fewer steps per trial)
        train_args = TrainingArguments(
            output_dir=f"training/outputs/hparam_trial_{trial.number}",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="no",
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=20 if val_dataset else None,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            bf16=True,
            max_seq_length=2048,  # rappresentativo del training reale (8192)
            report_to="none",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()

        # Get eval loss as metric
        if val_dataset:
            eval_loss = trainer.evaluate().get("eval_loss", float("inf"))
        else:
            eval_loss = trainer.state.log_history[-1].get("loss", float("inf"))

        # Clean up
        del model, trainer
        torch.cuda.empty_cache()

        print(f"[trial {trial.number}] eval_loss={eval_loss:.4f}")
        return eval_loss

    except Exception as e:
        print(f"[trial {trial.number}] FAILED: {e}")
        del model, trainer
        torch.cuda.empty_cache()
        return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/qlora_5060ti.yaml")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--study-name", default="fractalnova-hparam")
    ap.add_argument("--storage", help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=200, help="Limit training samples for speed")
    ap.add_argument("--max-eval-samples", type=int, default=50)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for hyperparameter tuning.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load tokenizer and subset of data
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": cfg["train_file"]}
    eval_file = cfg.get("eval_file")
    has_eval = bool(eval_file) and os.path.exists(eval_file)
    if has_eval:
        data_files["validation"] = eval_file

    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    val_ds = ds.get("validation", ds["train"]).select(range(min(args.max_eval_samples, len(ds.get("validation", ds["train"]))))) if has_eval else None

    print(f"[hparam] Data: {len(train_ds)} train, {len(val_ds) if val_ds else 0} val samples")
    print(f"[hparam] Search space: {len(SEARCH_SPACE)} hyperparameters")
    print(f"[hparam] Running {args.n_trials} trials...")

    # Create Optuna study
    storage = args.storage
    study_name = args.study_name

    if args.resume and storage:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"[hparam] Resumed study {study_name} with {len(study.trials)} existing trials")
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=args.resume,
            sampler=sampler,
            direction="minimize",
        )

    study.optimize(
        lambda trial: objective(trial, cfg, train_ds, val_ds, tokenizer),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best eval loss: {study.best_trial.value:.4f}")
    print(f"Best params:")
    for key, value in study.best_trial.params.items():
        if key == "lora_target_modules_count":
            print(f"  lora_target_modules: {TARGET_MODULE_OPTIONS[int(value)]}")
        else:
            print(f"  {key}: {value}")

    # Generate optimized config
    best = study.best_trial.params
    cfg["lora_r"] = best["lora_r"]
    cfg["lora_alpha"] = best["lora_alpha"]
    cfg["lora_dropout"] = best["lora_dropout"]
    cfg["learning_rate"] = best["learning_rate"]
    cfg["weight_decay"] = best["weight_decay"]
    cfg["warmup_ratio"] = best["warmup_ratio"]
    cfg["gradient_accumulation_steps"] = best["gradient_accumulation_steps"]
    cfg["lora_target_modules"] = TARGET_MODULE_OPTIONS[int(best["lora_target_modules_count"])]

    opt_path = "training/configs/qlora_optimized.yaml"
    with open(opt_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\nOptimized config written to {opt_path}")
    print(f"Train with: python training/train_qlora.py --config {opt_path}")


if __name__ == "__main__":
    main()
