"""
FractalNova · End-to-end training pipeline orchestrator.

Coordinates the complete model lifecycle:
  1. Dataset generation (synthetic via Gemini)
  2. Dataset preparation (chat format)
  3. Hyperparameter tuning (Optuna)
  4. Supervised fine-tuning (SFT)
  5. DPO alignment
  6. Evaluation & benchmarking
  7. Model merge & export
  8. Continuous retraining from production data

Usage:
    python training/pipeline.py --stage all [--quick]
    python training/pipeline.py --stage sft
    python training/pipeline.py --stage dpo
    python training/pipeline.py --stage benchmark
    python training/pipeline.py --full-cycle          # Complete production cycle

Config file: training/configs/pipeline.yaml (auto-created on first run)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen3-4B",
    "sft_config": "training/configs/qlora_5060ti.yaml",
    "dpo_config": "training/configs/dpo_5060ti.yaml",
    "optuna_config": "training/configs/qlora_5060ti.yaml",
    "data_dir": "training/data",
    "output_dir": "training/outputs",
    "sft_adapter_dir": "training/outputs/fractalnova-qlora",
    "dpo_adapter_dir": "training/outputs/fractalnova-dpo",
    "merged_model_dir": "training/outputs/fractalnova-pro-merged",
    "benchmark_report": "training/benchmark_report.json",
    "dataset_num_examples": 5000,
    "dataset_languages": "it,en,es,fr,de",
    "dpo_max_pairs": 1000,
    "n_hparam_trials": 30,
    "benchmark_models": "gemini-2.0-flash,fractalnova-pro,base-qwen",
    "trust_remote_code": False,
    "auto_deploy": False,
    "deploy_command": "kubectl set image deployment/fractalnova-vllm fractalnova-pro=ghcr.io/fractalnova/fractalnova-pro:latest",
}


def _ensure_config(path: str = "training/configs/pipeline.yaml") -> Dict:
    if not os.path.exists(path):
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
        print(f"[pipeline] Created default config at {path}")
        return dict(DEFAULT_CONFIG)
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cmd: str, stage: str, timeout: Optional[int] = None) -> bool:
    print(f"\n{'='*60}")
    print(f"[pipeline] STAGE: {stage}")
    print(f"[pipeline] Command: {cmd}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[pipeline] Stage {stage} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[pipeline] Stage {stage} failed: {e}")
        return False


def stage_generate_data(cfg: Dict) -> bool:
    langs = cfg.get("dataset_languages", "it,en,es,fr,de")
    n = cfg.get("dataset_num_examples", 5000)
    cmd = (
        f"python training/dataset_generator.py "
        f"--generate-all --num-per-task {max(100, n // 7)} "
        f"--languages {langs} "
        f"--out-dir {cfg['data_dir']}/generated"
    )
    return _run(cmd, "dataset_generation", timeout=7200)


def stage_prepare_data(cfg: Dict) -> bool:
    data_dir = cfg["data_dir"]
    generated = os.path.join(data_dir, "generated", "all_generated.jsonl")
    if not os.path.exists(generated):
        print(f"[pipeline] Generated data not found at {generated}, trying existing data...")
        inputs = " ".join([
            os.path.join(data_dir, "sample_books_it.jsonl"),
            os.path.join(data_dir, "sample_books_multi.jsonl"),
        ])
    else:
        inputs = generated

    cmd = (
        f"python training/prepare_dataset.py "
        f"--inputs {inputs} "
        f"--out-dir {data_dir} "
        f"--val-ratio 0.1"
    )
    return _run(cmd, "data_preparation", timeout=600)


def stage_hparam_tuning(cfg: Dict) -> bool:
    n = cfg.get("n_hparam_trials", 30)
    cmd = (
        f"python training/hyperparameter_tuning.py "
        f"--config {cfg['optuna_config']} "
        f"--n-trials {n} "
        f"--study-name fractalnova-v1"
    )
    return _run(cmd, "hyperparameter_tuning", timeout=86400)


def stage_sft(cfg: Dict) -> bool:
    config_path = cfg.get("sft_config", "training/configs/qlora_optimized.yaml")
    if not os.path.exists(config_path):
        config_path = "training/configs/qlora_5060ti.yaml"
        print(f"[pipeline] Optimized config not found, using {config_path}")
    cmd = f"python training/train_qlora.py --config {config_path}"
    return _run(cmd, "supervised_fine_tuning", timeout=86400)


def stage_generate_dpo_pairs(cfg: Dict) -> bool:
    data_path = os.path.join(cfg["data_dir"], "train.jsonl")
    if not os.path.exists(data_path):
        print(f"[pipeline] Training data not found at {data_path}")
        return False
    cmd = (
        f"python training/train_dpo.py "
        f"--generate-pairs --base-data {data_path} "
        f"--max-pairs {cfg.get('dpo_max_pairs', 1000)}"
    )
    return _run(cmd, "dpo_pair_generation", timeout=3600)


def stage_dpo(cfg: Dict) -> bool:
    cmd = (
        f"python training/train_dpo.py "
        f"--base-model {cfg['base_model']} "
        f"--adapter {cfg['sft_adapter_dir']} "
        f"--dpo-data training/data/dpo_pairs.jsonl "
        f"--output {cfg['dpo_adapter_dir']}"
    )
    return _run(cmd, "dpo_alignment", timeout=86400)


def stage_merge(cfg: Dict) -> bool:
    adapter = cfg.get("dpo_adapter_dir")
    if not os.path.exists(adapter):
        adapter = cfg.get("sft_adapter_dir")
        print(f"[pipeline] DPO adapter not found, using SFT adapter: {adapter}")
    cmd = (
        f"python training/merge_and_export.py "
        f"--base {cfg['base_model']} "
        f"--adapter {adapter} "
        f"--out {cfg['merged_model_dir']}"
    )
    return _run(cmd, "model_merge", timeout=7200)


def stage_evaluate(cfg: Dict) -> bool:
    adapter = cfg.get("dpo_adapter_dir")
    if not os.path.exists(adapter):
        adapter = cfg.get("sft_adapter_dir")
    cmd = (
        f"python training/evaluate.py "
        f"--base {cfg['base_model']} "
        f"--adapter {adapter} "
        f"--load-4bit"
    )
    return _run(cmd, "evaluation", timeout=3600)


def stage_benchmark(cfg: Dict) -> bool:
    models = cfg.get("benchmark_models", "gemini-2.0-flash,fractalnova-pro,base-qwen")
    cmd = (
        f"python training/benchmark.py "
        f"--models {models} "
        f"--tasks write_incipit,humanize_prose,write_synopsis "
        f"--languages it,en "
        f"--output {cfg['benchmark_report']}"
    )
    return _run(cmd, "benchmarking", timeout=3600)


def stage_continuous(cfg: Dict) -> bool:
    cmd = (
        f"python training/train_continuous.py "
        f"--pipeline "
        f"--base-model {cfg['base_model']} "
        f"--adapter {cfg['merged_model_dir']}"
    )
    return _run(cmd, "continuous_training", timeout=3600)


STAGES = {
    "generate_data": stage_generate_data,
    "prepare_data": stage_prepare_data,
    "hparam_tuning": stage_hparam_tuning,
    "sft": stage_sft,
    "generate_dpo_pairs": stage_generate_dpo_pairs,
    "dpo": stage_dpo,
    "merge": stage_merge,
    "evaluate": stage_evaluate,
    "benchmark": stage_benchmark,
    "continuous": stage_continuous,
}

STAGE_ORDER = [
    "generate_data",
    "prepare_data",
    "hparam_tuning",
    "sft",
    "generate_dpo_pairs",
    "dpo",
    "merge",
    "evaluate",
    "benchmark",
]

STAGE_DESCRIPTIONS = {
    "generate_data": "Generate synthetic training data via Gemini API",
    "prepare_data": "Convert raw data to chat format for SFT",
    "hparam_tuning": "Optuna hyperparameter search",
    "sft": "Supervised fine-tuning (QLoRA)",
    "generate_dpo_pairs": "Create preference pairs for DPO",
    "dpo": "Direct Preference Optimization alignment",
    "merge": "Merge LoRA adapter into base model",
    "evaluate": "Perplexity + qualitative evaluation",
    "benchmark": "Benchmark against GPT-4/Claude/Gemini",
    "continuous": "Continuous retraining from production data",
}


def run_stage(stage: str, cfg: Dict) -> bool:
    fn = STAGES.get(stage)
    if not fn:
        print(f"[pipeline] Unknown stage: {stage}")
        return False
    print(f"\n{'#'*60}")
    print(f"# STAGE: {stage}")
    print(f"# {STAGE_DESCRIPTIONS.get(stage, '')}")
    print(f"# Started: {datetime.utcnow().isoformat()}")
    print(f"{'#'*60}\n")
    start = time.time()
    success = fn(cfg)
    elapsed = time.time() - start
    status = "PASSED" if success else "FAILED"
    print(f"\n{'#'*60}")
    print(f"# STAGE {stage}: {status} ({elapsed:.0f}s)")
    print(f"{'#'*60}\n")
    return success


def run_quick(cfg: Dict) -> bool:
    """Quick cycle: prepare_data -> sft -> merge -> evaluate"""
    print("\n[quick] Running quick training cycle...")
    for stage in ["prepare_data", "sft", "merge", "evaluate"]:
        if not run_stage(stage, cfg):
            print(f"[quick] Failed at stage {stage}")
            return False
    print("[quick] Quick cycle complete!")
    return True


def run_full_cycle(cfg: Dict) -> bool:
    """Full production cycle"""
    print(f"\n{'='*60}")
    print(f"FULL TRAINING CYCLE")
    print(f"Base model: {cfg['base_model']}")
    print(f"Output: {cfg['merged_model_dir']}")
    print(f"{'='*60}")

    report = {
        "started_at": datetime.utcnow().isoformat(),
        "stages": [],
        "final_status": "unknown",
    }

    for stage in STAGE_ORDER:
        start = time.time()
        success = run_stage(stage, cfg)
        elapsed = time.time() - start
        report["stages"].append({"stage": stage, "success": success, "elapsed_s": elapsed})
        if not success:
            report["final_status"] = f"failed_at_{stage}"
            break

    report["completed_at"] = datetime.utcnow().isoformat()
    report["final_status"] = report.get("final_status") or "completed"

    report_path = f"training/pipeline_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total = sum(s["elapsed_s"] for s in report["stages"])
    print(f"\n{'='*60}")
    print(f"PIPELINE {report['final_status'].upper()}")
    print(f"Total time: {total/3600:.1f}h")
    print(f"Report: {report_path}")
    print(f"{'='*60}")

    if report["final_status"] == "completed" and cfg.get("auto_deploy"):
        print("[pipeline] Auto-deploy enabled. Deploying merged model...")
        _run(cfg["deploy_command"], "deploy", timeout=300)

    return report["final_status"] == "completed"


def main():
    ap = argparse.ArgumentParser(description="FractalNova training pipeline")
    ap.add_argument("--stage", choices=list(STAGES.keys()) + ["all", "quick", "full-cycle"],
                    default="sft", help="Stage to run")
    ap.add_argument("--quick", action="store_true", help="Quick training cycle (SFT -> merge -> eval)")
    ap.add_argument("--full-cycle", action="store_true", help="Complete production training cycle")
    ap.add_argument("--config", default="training/configs/pipeline.yaml")
    ap.add_argument("--list-stages", action="store_true", help="List available stages")
    args = ap.parse_args()

    cfg = _ensure_config(args.config)

    if args.list_stages:
        print("\nAvailable stages:")
        for s in STAGE_ORDER:
            print(f"  {s:25s} {STAGE_DESCRIPTIONS.get(s, '')}")
        print(f"  {'all':25s} Run all stages sequentially")
        print(f"  {'quick':25s} Quick cycle (prepare->sft->merge->eval)")
        print(f"  {'full-cycle':25s} Complete production cycle")
        return

    if args.full_cycle:
        run_full_cycle(cfg)
        return

    if args.quick:
        run_quick(cfg)
        return

    if args.stage == "all":
        for s in STAGE_ORDER:
            if not run_stage(s, cfg):
                print(f"[pipeline] Pipeline failed at {s}")
                break
        return

    run_stage(args.stage, cfg)


if __name__ == "__main__":
    main()
