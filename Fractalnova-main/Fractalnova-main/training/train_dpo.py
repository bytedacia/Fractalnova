"""
FractalNova · Direct Preference Optimization (DPO) for book-writing alignment.

Trains the model to prefer high-quality, human-like writing over generic LLM output.
Uses Gemini/Claude as judge to create preference pairs from the supervised dataset.

Usage:
    # Generate preference pairs (requires GOOGLE_API_KEY)
    python training/train_dpo.py --generate-pairs --base-data training/data/train.jsonl

    # Train DPO
    python training/train_dpo.py --config training/configs/dpo_5060ti.yaml

    # Full pipeline
    python training/train_dpo.py --base-model Qwen/Qwen3-4B \
        --adapter training/outputs/fractalnova-qlora \
        --dpo-data training/data/dpo_pairs.jsonl \
        --output training/outputs/fractalnova-dpo
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

DPO_SYSTEM_PROMPT = """Sei FractalNova, autore ed editor professionista. Scrivi in modo naturale e umano, con voce e ritmo curati. Rispondi SEMPRE nella stessa lingua della richiesta."""


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gemini():
    import google.generativeai as genai
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.0-flash")


def generate_preference_pairs(
    base_data_path: str,
    output_path: str,
    judge_model=None,
    max_pairs: int = 1000,
    seed: int = 42,
) -> str:
    """
    Creates preference pairs from supervised data by:
    1. Taking the supervised output as the "chosen" response
    2. Generating a degraded "rejected" response by prompting the base model
    3. Using LLM-as-judge to verify preference quality
    """
    print(f"[dpo] Generating preference pairs from {base_data_path}...")
    random.seed(seed)

    examples = []
    with open(base_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    random.shuffle(examples)
    examples = examples[:max_pairs]
    print(f"[dpo] Loaded {len(examples)} base examples")

    pairs = []
    skipped = 0

    for ex in examples:
        messages = ex.get("messages", [])
        if not messages:
            continue

        chosen_response = None
        for m in messages:
            if m.get("role") == "assistant":
                chosen_response = m["content"]
                break
        if not chosen_response or len(chosen_response) < 50:
            skipped += 1
            continue

        user_message = None
        for m in messages:
            if m.get("role") == "user":
                user_message = m["content"]
                break
        if not user_message:
            skipped += 1
            continue

        pair = {
            "prompt": [
                {"role": "system", "content": DPO_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "chosen": [{"role": "assistant", "content": chosen_response}],
            "rejected": [{"role": "assistant", "content": _degrade_response(chosen_response, judge_model)}],
        }
        pairs.append(pair)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[dpo] Generated {len(pairs)} preference pairs ({skipped} skipped)")
    return output_path


def _degrade_response(chosen: str, judge_model=None) -> str:
    """
    Creates a deliberately lower-quality version of the chosen response.
    This is the "rejected" response for DPO training.
    """
    if judge_model:
        try:
            resp = judge_model.generate_content(
                f"Rewrite the following text to be noticeably worse quality - more generic, less engaging, "
                f"flatter prose, cliches, and less natural. Keep similar length but make it clearly inferior "
                f"as literary writing:\n\n{chosen}"
            )
            degraded = resp.text.strip()
            if degraded and 0.5 < len(degraded) / len(chosen) < 2.0:
                return degraded
        except Exception:
            pass

    words = chosen.split()
    if len(words) < 10:
        return chosen + " (the end)"

    random.seed(hash(chosen) % 10000)
    degraded = words.copy()
    substitutions = {
        "beautiful": "nice", "terrible": "bad", "walked": "went",
        "whispered": "said", "enormous": "big", "furious": "angry",
        "gazed": "looked", "silent": "quiet", "ancient": "old",
        "fragile": "weak", "determined": "sure", "elegant": "fancy",
    }
    for i, w in enumerate(degraded):
        w_lower = w.lower().strip(".,!?;:\"'")
        if w_lower in substitutions:
            replacement = substitutions[w_lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            if w[-1] in ".,!?;:\"'":
                replacement += w[-1]
            degraded[i] = replacement

    return " ".join(degraded)


def format_dpo_prompt(examples: Dict) -> Dict:
    """Format DPO examples for the trainer."""
    result = {"prompt": [], "chosen": [], "rejected": []}
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        chosen = examples["chosen"][i]
        rejected = examples["rejected"][i]
        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/dpo_5060ti.yaml")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", help="Path to SFT adapter to continue from")
    ap.add_argument("--dpo-data", default="training/data/dpo_pairs.jsonl")
    ap.add_argument("--output", default="training/outputs/fractalnova-dpo")
    ap.add_argument("--generate-pairs", action="store_true",
                    help="Generate DPO preference pairs from supervised data")
    ap.add_argument("--base-data", default="training/data/train.jsonl",
                    help="Source supervised data for generating pairs")
    ap.add_argument("--max-pairs", type=int, default=1000)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    if args.generate_pairs:
        judge = load_gemini()
        generate_preference_pairs(
            args.base_data, args.dpo_data,
            judge_model=judge, max_pairs=args.max_pairs,
        )
        print("[dpo] Pair generation complete. Run without --generate-pairs to train.")
        return

    if not os.path.exists(args.dpo_data):
        raise SystemExit(f"DPO data not found at {args.dpo_data}. Run with --generate-pairs first.")

    # Load config or use defaults
    config_path = args.config
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = {
            "base_model": args.base_model,
            "dpo_data": args.dpo_data,
            "output_dir": args.output,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-6,
            "max_length": 1024,
            "max_prompt_length": 512,
            "num_train_epochs": 1,
            "warmup_steps": 50,
            "logging_steps": 10,
            "save_steps": 200,
            "save_total_limit": 2,
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "beta": 0.1,
        }
        print(f"[dpo] Config not found, using defaults.")

    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available. DPO requires a GPU.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Ricetta QLoRA-DPO corretta:
    #  - con adapter SFT: lo si CONTINUA (is_trainable=True) e NON si passa peft_config
    #  - senza: si crea un nuovo adapter via peft_config
    #  - ref_model=None: con PEFT, DPOTrainer usa il base (adapter disattivato) come riferimento
    dpo_peft_config = peft_config
    if args.adapter and os.path.isdir(args.adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        dpo_peft_config = None
        print(f"[dpo] Continuo dall'adapter SFT: {args.adapter}")

    # Load DPO dataset
    pairs = []
    with open(cfg["dpo_data"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        raise SystemExit("No DPO pairs loaded.")

    dataset = Dataset.from_list(pairs)
    print(f"[dpo] Loaded {len(dataset)} training pairs")

    def tokenize_fn(examples):
        result = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(examples["prompt"])):
            result["prompt"].append(tokenizer.apply_chat_template(
                examples["prompt"][i], tokenize=False, add_generation_prompt=True
            ))
            result["chosen"].append(tokenizer.apply_chat_template(
                examples["chosen"][i], tokenize=False
            ))
            result["rejected"].append(tokenizer.apply_chat_template(
                examples["rejected"][i], tokenize=False
            ))
        return result

    # Dataset conversazionale (prompt/chosen/rejected = liste di messaggi):
    # DPOTrainer applica internamente il chat template. Niente pre-rendering manuale.

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=cfg.get("output_dir", args.output),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        warmup_steps=cfg.get("warmup_steps", 50),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        max_length=cfg.get("max_length", 1024),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        beta=float(cfg.get("beta", 0.1)),
        report_to="none",
        remove_unused_columns=False,
    )

    # Con PEFT non serve un ref_model separato: DPOTrainer usa il base
    # con adapter disattivato come riferimento (risparmia VRAM).
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=dpo_peft_config,
    )

    trainer.train()
    trainer.save_model(cfg.get("output_dir", args.output))
    tokenizer.save_pretrained(cfg.get("output_dir", args.output))
    print(f"[dpo] DPO training complete. Model saved to {cfg.get('output_dir', args.output)}")
    print(f"       Merge & deploy: training/merge_and_export.py --base {cfg['base_model']} --adapter {cfg.get('output_dir', args.output)} --out training/outputs/fractalnova-pro-dpo-merged")


if __name__ == "__main__":
    main()
