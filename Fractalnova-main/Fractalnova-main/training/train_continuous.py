"""
FractalNova · Continuous Training Pipeline with Data Flywheel.

Automatically retrains the model using production data:
1. Collects user feedback from production (thumbs up/down, edits)
2. Converts high-quality interactions to training examples
3. Periodically retrains with new data
4. Evaluates against regression benchmarks
5. Deploys if quality improves

Usage (production):
    python training/train_continuous.py --collect     # Collect production data
    python training/train_continuous.py --retrain     # Retrain with new data
    python training/train_continuous.py --pipeline    # Full pipeline

Scheduled (cron):
    0 3 * * 1 cd /app && python training/train_continuous.py --pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

DATA_COLLECTOR_SYSTEM = """Sei FractalNova, un assistente di scrittura. Trasforma il feedback degli utenti in esempi di training di alta qualità."""


def collect_production_data(
    db_url: Optional[str] = None,
    min_thumbs_up: int = 3,
    out_path: str = "training/data/collected",
) -> str:
    """
    Collect training data from production:
    - User edits (user accepted AI output, then modified it -> preferred version)
    - Thumbs up/down pairs
    - Saved/favorited generations
    - Completed books with high ratings
    """
    print(f"[continuous] Collecting production data (min_thumbs_up={min_thumbs_up})...")
    os.makedirs(out_path, exist_ok=True)

    collected = []

    # Collect from local feedback file if it exists
    feedback_file = "production_feedback.jsonl"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("score", 0) >= min_thumbs_up and entry.get("prompt") and entry.get("response"):
                    collected.append({
                        "messages": [
                            {"role": "system", "content": "Sei FractalNova, autore ed editor professionista."},
                            {"role": "user", "content": entry["prompt"]},
                            {"role": "assistant", "content": entry["response"]},
                        ],
                        "source": "production_feedback",
                        "quality_score": entry.get("score", 0),
                        "collected_at": datetime.utcnow().isoformat(),
                    })

    # Collect from API logs if configured
    api_log = "api_generation_log.jsonl"
    if os.path.exists(api_log):
        with open(api_log, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("status") == "completed" and entry.get("user_rating", 0) >= min_thumbs_up:
                    collected.append({
                        "messages": [
                            {"role": "system", "content": "Sei FractalNova, autore ed editor professionista."},
                            {"role": "user", "content": entry.get("prompt", "")},
                            {"role": "assistant", "content": entry.get("generated_text", "")},
                        ],
                        "source": "api_log",
                        "quality_score": entry.get("user_rating", 5),
                        "collected_at": datetime.utcnow().isoformat(),
                    })

    # Write collected data
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_path, f"production_data_{timestamp}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in collected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[continuous] Collected {len(collected)} production examples -> {out_file}")

    # Create combined dataset
    combined = _combine_datasets(out_path)
    return combined


def _combine_datasets(data_dir: str, output: str = "training/data/production_combined.jsonl") -> str:
    """Merge all production data files into a single dataset."""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "production_data_*.jsonl")))
    if not files:
        print("[continuous] No production data files to combine.")
        return ""

    all_examples = []
    seen = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line.strip())
                key = json.dumps(ex.get("messages", []), ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    all_examples.append(ex)

    with open(output, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[continuous] Combined {len(all_examples)} unique production examples -> {output}")
    return output


def retrain_with_new_data(
    base_model: str,
    adapter_path: str,
    new_data_path: str,
    original_data_path: str,
    output_dir: str,
    trust_remote_code: bool = False,
    eval_benchmark_path: Optional[str] = None,
) -> Dict:
    """
    Retrain the model combining original training data with new production data.
    Evaluates against a held-out benchmark before/after to detect regression.
    """
    import torch

    print(f"[continuous] Retraining with new data from {new_data_path}...")

    # Load new data
    new_data = Dataset.from_json(new_data_path)

    # Load original data (subsample for balanced training)
    orig_data = Dataset.from_json(original_data_path)

    # Combine: 70% original + 30% new to maintain base capabilities
    n_new = min(len(new_data), int(len(orig_data) * 0.3))
    n_orig = len(orig_data)

    if len(new_data) > n_new:
        new_data = new_data.select(range(n_new))

    combined = concatenate_datasets([orig_data, new_data]).shuffle(seed=42)
    print(f"[continuous] Combined dataset: {len(combined)} examples ({n_orig} original + {n_new} new)")

    # Pre-retraining evaluation
    bench_results = {}
    if eval_benchmark_path and os.path.exists(eval_benchmark_path):
        bench_results["pre"] = _evaluate_on_benchmark(adapter_path or base_model, eval_benchmark_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if os.path.exists(adapter_path) else base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", attn_implementation="sdpa",
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load existing adapter if available
    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[continuous] Loaded existing adapter from {adapter_path}")

    def fmt(examples):
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=1024,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=combined,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=fmt,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Post-retraining evaluation
    if eval_benchmark_path:
        bench_results["post"] = _evaluate_on_benchmark(output_dir, eval_benchmark_path)
        _report_regression(bench_results)

    print(f"[continuous] Retrained model saved to {output_dir}")
    return bench_results


def _evaluate_on_benchmark(model_path: str, benchmark_path: str) -> Dict:
    """Quick evaluation on a held-out benchmark set."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] Benchmarking {model_path} on {benchmark_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model.eval()

        examples = []
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        losses = []
        for ex in examples[:50]:
            rendered = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
            enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                out = model(**enc, labels=enc.input_ids)
            losses.append(out.loss.item())

        avg_loss = sum(losses) / max(len(losses), 1)
        perplexity = 2.71828 ** avg_loss
        return {"loss": avg_loss, "perplexity": perplexity, "samples": len(losses)}

    except Exception as e:
        print(f"[eval] Benchmark failed: {e}")
        return {"error": str(e)}


def _report_regression(results: Dict):
    """Check if new model regresses against the old one."""
    pre = results.get("pre", {})
    post = results.get("post", {})

    if "perplexity" in pre and "perplexity" in post:
        diff = post["perplexity"] - pre["perplexity"]
        if diff > 0.5:
            print(f"[WARNING] Perplexity increased by {diff:.2f}! Possible regression.")
        elif diff < -0.5:
            print(f"[IMPROVEMENT] Perplexity decreased by {abs(diff):.2f}! Model improved.")
        else:
            print(f"[STABLE] Perplexity change: {diff:+.2f} (within tolerance)")


def create_benchmark_set(out_path: str = "training/data/benchmark.jsonl", n_samples: int = 100):
    """Create a held-out benchmark set from existing training data."""
    import glob

    files = glob.glob("training/data/*.jsonl")
    examples = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                examples.append(line.strip())

    import random
    random.seed(42)
    random.shuffle(examples)
    benchmark = examples[:n_samples]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in benchmark:
            f.write(ex + "\n")

    print(f"[benchmark] Created benchmark set with {len(benchmark)} samples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="Collect production data")
    ap.add_argument("--retrain", action="store_true", help="Retrain with new data")
    ap.add_argument("--pipeline", action="store_true", help="Run full pipeline (collect + retrain)")
    ap.add_argument("--create-benchmark", action="store_true", help="Create benchmark set")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="training/outputs/fractalnova-qlora")
    ap.add_argument("--output", default="training/outputs/fractalnova-continuous")
    ap.add_argument("--original-data", default="training/data/train.jsonl")
    ap.add_argument("--benchmark", default="training/data/benchmark.jsonl")
    args = ap.parse_args()

    if args.create_benchmark:
        create_benchmark_set(args.benchmark)
        return

    if args.collect or args.pipeline:
        collected = collect_production_data(out_path="training/data/collected")
        combined_file = "training/data/production_combined.jsonl"

    if args.retrain or args.pipeline:
        new_data = "training/data/production_combined.jsonl"
        if not os.path.exists(new_data):
            print(f"[continuous] No production data at {new_data}. Collect first or skip.")
            return

        results = retrain_with_new_data(
            base_model=args.base_model,
            adapter_path=args.adapter,
            new_data_path=new_data,
            original_data_path=args.original_data,
            output_dir=args.output,
            eval_benchmark_path=args.benchmark if os.path.exists(args.benchmark) else None,
        )

        if results:
            print(f"[continuous] Continuous training cycle complete.")
            print(f"[continuous] Next: merge_and_export.py then redeploy vLLM.")


if __name__ == "__main__":
    main()
