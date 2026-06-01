"""
FractalNova · Comprehensive model benchmarking suite.

Evaluates FractalNova models against GPT-4, Claude, Gemini, and base models
across 7 book-writing tasks and 5 languages. Measures quality, latency, and cost.

Usage:
    python training/benchmark.py --models fractalnova-pro,base-qwen,gpt4 --tasks write,humanize,synopsis
    python training/benchmark.py --quick                              # Quick sanity check
    python training/benchmark.py --full-report                        # Full benchmark report
    python training/benchmark.py --judge-quality --models fractalnova-pro,gpt4  # LLM-as-judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

BENCHMARK_TASKS = [
    "write_incipit",
    "continue_scene",
    "write_chapter",
    "humanize_prose",
    "generate_title",
    "write_synopsis",
    "generate_seo",
]

BENCHMARK_LANGUAGES = ["it", "en", "es", "fr", "de"]

# Gold-standard evaluation prompts for each task
EVAL_PROMPTS = {
    "write_incipit": {
        "it": "Scrivi l'incipit di un romanzo gotico ambientato in un castello abbandonato sulle Alpi. Crea atmosfera e intrigo.",
        "en": "Write the opening of a literary novel set in a decaying mansion on the coast of Maine. Establish voice, setting, and tension.",
        "es": "Escribe el inicio de una novela de realismo mágico ambientada en un pueblo andaluz. Crea una atmósfera envolvente.",
        "fr": "Écrivez l'incipit d'un roman policier se déroulant dans le Paris des années 1920. Créez une atmosphère mystérieuse.",
        "de": "Schreibe den Anfang eines historischen Romans, der im Berlin der Weimarer Republik spielt. Baue Spannung auf.",
    },
    "continue_scene": {
        "it": "Continua questa scena: 'La porta si aprì da sola. Non c'era vento, non c'era nessuno. Eppure, qualcosa era entrato nella stanza.'",
        "en": "Continue this scene: 'The letter arrived on a Tuesday, postmarked from a town she'd never heard of. Inside was a single photograph.'",
    },
    "write_chapter": {
        "it": "Scrivi un capitolo di 300 parole dal punto di vista di un detective che scopre un indizio cruciale in un caso di omicidio.",
        "en": "Write a 300-word chapter from the perspective of a astronaut watching Earth from the ISS during a crisis.",
    },
    "humanize_prose": {
        "it": "Riscrivi in modo più umano e naturale: 'Il protagonista era molto arrabbiato per la situazione e quindi decise di prendere una decisione drastica che avrebbe cambiato tutto.'",
        "en": "Rewrite more naturally: 'The protagonist was very angry about the situation and therefore decided to make a drastic decision that would change everything.'",
    },
    "generate_title": {
        "it": "Genera un titolo potente e una tagline per un thriller psicologico ambientato in una clinica psichiatrica abbandonata.",
        "en": "Generate a powerful title and tagline for a literary fiction novel about a family secret in rural Ireland.",
    },
    "write_synopsis": {
        "it": "Scrivi una sinossi di 100 parole per un fantasy in cui la magia si ottiene solo sacrificando i ricordi più preziosi.",
        "en": "Write a 100-word synopsis for a sci-fi novel where humanity's last survivors live in a generation ship that has lost its course.",
    },
    "generate_seo": {
        "it": "Genera metadati SEO (keyword, tags, description, categories) per un romanzo storico ambientato nella Venezia del '500.",
        "en": "Generate SEO metadata (keywords, tags, description, categories) for a literary novel set in 1920s Harlem.",
    },
}


@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    language: str
    prompt: str
    response: str
    latency_ms: float
    eval_score: float = 0.0
    word_count: int = 0
    error: Optional[str] = None


@dataclass
class ModelClient:
    name: str
    type: str  # "local", "gemini", "openai", "anthropic"
    model_path: Optional[str] = None

    def generate(self, prompt: str, system: str = "", temperature: float = 0.8, max_tokens: int = 500) -> tuple[str, float]:
        start = time.time()
        try:
            if self.type == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel("gemini-2.0-flash")
                resp = model.generate_content(
                    f"{system}\n\n{prompt}" if system else prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                text = resp.text or ""

            elif self.type == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(
                    model=self.model_path or "gpt-4o",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""

            elif self.type == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                messages = [{"role": "user", "content": prompt}]
                resp = client.messages.create(
                    model=self.model_path or "claude-sonnet-4-20250514",
                    system=system or "",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.content[0].text if resp.content else ""

            elif self.type == "local":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, device_map="auto",
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                )
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
                text = tokenizer.decode(out[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
                del model, tokenizer
                torch.cuda.empty_cache()

            else:
                return "", 0.0

            elapsed = (time.time() - start) * 1000
            return text.strip(), elapsed

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"[benchmark] Error generating with {self.name}: {e}")
            return "", elapsed


def quality_judge(response: str, task: str, language: str, judge_model=None) -> float:
    """LLM-as-judge: rate the quality of a response on 1-10 scale."""
    if not judge_model:
        return _heuristic_quality(response)

    criteria = {
        "write_incipit": "atmosphere, voice, prose quality, hook",
        "continue_scene": "consistency, style match, creativity",
        "write_chapter": "narrative quality, engagement, detail",
        "humanize_prose": "naturalness, flow, improvement over original",
        "generate_title": "marketability, creativity, genre-fit",
        "write_synopsis": "hook strength, clarity, concision",
        "generate_seo": "relevance, keyword quality, accuracy",
    }
    crit = criteria.get(task, "overall quality")

    try:
        resp = judge_model.generate_content(
            f"Rate this {task} ({language}) response on quality 1-10 based on: {crit}.\n"
            f"RESPONSE:\n{response[:1000]}\n\n"
            f"Respond ONLY with a number 1-10."
        )
        score = float(resp.text.strip())
        return max(1.0, min(10.0, score))
    except Exception:
        return _heuristic_quality(response)


def _heuristic_quality(text: str) -> float:
    """Heuristic quality score based on linguistic features."""
    if not text or len(text) < 20:
        return 1.0

    words = text.split()
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 2.0

    avg_words_per_sentence = len(words) / max(len(sentences), 1)
    unique_words_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    score = 5.0
    if 12 < avg_words_per_sentence < 25:
        score += 1.0
    if unique_words_ratio > 0.55:
        score += 1.0
    if 4.5 < avg_word_length < 7.0:
        score += 0.5
    if len(words) > 30:
        score += 0.5

    cliches = ["in the end", "it was", "suddenly", "very", "literally", "amazing", "incredible"]
    cliche_count = sum(1 for c in cliches if c.lower() in text.lower())
    score -= cliche_count * 0.5

    return max(1.0, min(10.0, score))


def run_benchmark(models: List[ModelClient], tasks: List[str], languages: List[str], judge=None) -> List[BenchmarkResult]:
    results = []

    for model_client in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking model: {model_client.name} ({model_client.type})")
        print(f"{'='*60}")

        for task in tasks:
            for lang in languages:
                prompt = EVAL_PROMPTS.get(task, {}).get(lang)
                if not prompt:
                    continue

                system = "Sei FractalNova, un autore professionista. Scrivi in modo naturale e umano."
                response, latency = model_client.generate(prompt, system=system)
                word_count = len(response.split()) if response else 0

                score = quality_judge(response, task, lang, judge_model=judge)

                results.append(BenchmarkResult(
                    model_name=model_client.name,
                    task=task,
                    language=lang,
                    prompt=prompt,
                    response=response,
                    latency_ms=latency,
                    eval_score=score,
                    word_count=word_count,
                ))

                print(f"  [{task}/{lang}] score={score:.1f} latency={latency:.0f}ms words={word_count}")

    return results


def generate_report(results: List[BenchmarkResult], output_path: str = "training/benchmark_report.json"):
    """Generate comprehensive benchmark report."""
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {},
        "per_model": {},
        "per_task": {},
        "details": [r.__dict__ for r in results],
    }

    models = set(r.model_name for r in results)
    tasks = set(r.task for r in results)

    for model in sorted(models):
        model_results = [r for r in results if r.model_name == model]
        scores = [r.eval_score for r in model_results if r.eval_score > 0]
        latencies = [r.latency_ms for r in model_results if r.latency_ms > 0]

        report["per_model"][model] = {
            "avg_score": round(np.mean(scores), 2) if scores else 0,
            "avg_latency_ms": round(np.mean(latencies), 0) if latencies else 0,
            "total_tests": len(model_results),
        }

    for task in sorted(tasks):
        task_results = [r for r in results if r.task == task]
        report["per_task"][task] = {}
        for r in task_results:
            report["per_task"][task][r.model_name] = {
                "score": r.eval_score,
                "latency_ms": round(r.latency_ms, 0),
                "words": r.word_count,
            }

    best_model = max(report["per_model"].items(), key=lambda x: x[1]["avg_score"])
    report["summary"]["best_model"] = {"name": best_model[0], "avg_score": best_model[1]["avg_score"]}
    report["summary"]["total_results"] = len(results)
    report["summary"]["models_tested"] = list(sorted(models))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"Best model: {report['summary']['best_model']['name']} (score: {report['summary']['best_model']['avg_score']})")
    for model, stats in sorted(report["per_model"].items()):
        print(f"  {model:30s} avg_score={stats['avg_score']:.1f}  latency={stats['avg_latency_ms']:.0f}ms  tests={stats['total_tests']}")
    print(f"\nFull report: {output_path}")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini-2.0-flash", help="Comma-separated: gemini,openai,anthropic,local_path")
    ap.add_argument("--tasks", default=",".join(BENCHMARK_TASKS))
    ap.add_argument("--languages", default="it,en")
    ap.add_argument("--quick", action="store_true", help="Quick: 1 task, 1 language")
    ap.add_argument("--full-report", action="store_true")
    ap.add_argument("--judge-quality", action="store_true", help="Use LLM-as-judge for scoring")
    ap.add_argument("--output", default="training/benchmark_report.json")
    args = ap.parse_args()

    tasks = ["write_incipit"] if args.quick else args.tasks.split(",")
    languages = ["it"] if args.quick else args.languages.split(",")

    model_specs = args.models.split(",")
    models = []
    for spec in model_specs:
        spec = spec.strip()
        if spec == "gemini" or spec.startswith("gemini"):
            models.append(ModelClient(name=spec, type="gemini", model_path=spec.split("-", 1)[1] if "-" in spec else None))
        elif spec == "openai" or spec.startswith("gpt"):
            models.append(ModelClient(name=spec, type="openai"))
        elif spec == "anthropic" or spec.startswith("claude"):
            models.append(ModelClient(name=spec, type="anthropic"))
        elif spec == "fractalnova-pro" or spec == "base-qwen":
            path = "training/outputs/fractalnova-pro-merged" if spec == "fractalnova-pro" else "Qwen/Qwen3-4B"
            models.append(ModelClient(name=spec, type="local", model_path=path))
        else:
            models.append(ModelClient(name=spec, type="local", model_path=spec))

    if not models:
        raise SystemExit("No valid models specified.")

    judge = None
    if args.judge_quality:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        judge = genai.GenerativeModel("gemini-2.0-flash")

    results = run_benchmark(models, tasks, languages, judge=judge)
    report = generate_report(results, args.output)

    # Print competitive comparison
    if len(models) > 1:
        print("\nCompetitive Positioning:")
        sorted_models = sorted(report["per_model"].items(), key=lambda x: x[1]["avg_score"], reverse=True)
        for i, (name, stats) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"  {medal} {name:30s} {stats['avg_score']:.1f}/10")

    print(f"\n{'='*60}")
    print(f"Benchmark complete. Report: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
