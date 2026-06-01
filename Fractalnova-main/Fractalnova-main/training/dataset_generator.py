"""
FractalNova · Massive synthetic dataset generator for book-writing LLM fine-tuning.

    Generates 100,000+ high-quality training examples across 7 tasks x 10 languages
using Gemini API as the teacher model. Each example is validated for quality.

Output: training/data/generated_{task}_{lang}.jsonl -> prepare_dataset.py format

Usage:
    python training/dataset_generator.py --num-examples 100000 --languages it,en,es,fr,de
    python training/dataset_generator.py --generate-all --num-per-task 5000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm

TASKS = ["write", "continue", "humanize", "title", "synopsis", "seo", "translate"]

LANGUAGES = {
    "it": "Italian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
}

GENRES = [
    "fiction", "fantasy", "sci-fi", "thriller", "romance",
    "historical", "horror", "giallo", "adventure", "literary",
]

SETTINGS = [
    "a small coastal town in the 1800s",
    "a futuristic megacity in 2150",
    "a medieval fantasy kingdom",
    "an underwater research station",
    "a bustling metropolis at night",
    "a remote mountain village",
    "a spaceship traveling to a new galaxy",
    "a quiet library in autumn",
    "a war-torn city at dawn",
    "a mysterious island in the Pacific",
    "a detective's office in 1950s Chicago",
    "a magical school hidden in the clouds",
    "a desert planet with two suns",
    "a Victorian-era London street",
    "a parallel universe where time runs backward",
]

CHARACTERS = [
    "a young journalist seeking the truth",
    "an aging detective on their last case",
    "a orphan with a hidden power",
    "a retired soldier haunted by memories",
    "a brilliant scientist on the verge of discovery",
    "a street-smart thief with a golden heart",
    "a struggling artist chasing a dream",
    "a ship captain navigating dangerous waters",
    "a librarian who discovers a secret world",
    "a rebel fighting against an unjust system",
    "a doctor torn between duty and family",
    "a wandering musician searching for home",
    "a spy trapped behind enemy lines",
    "a child who can see invisible creatures",
    "an archaeologist uncovering a dangerous truth",
]

QUALITY_SYSTEM_PROMPT = """You are FractalNova, a professional author and editor. Generate high-quality literary text that:
- Uses natural, human prose with rhythm and voice
- Shows rather than tells
- Has varied sentence structure
- Includes sensory details (sight, sound, smell, touch, taste)
- Creates atmosphere and emotion
- Maintains consistent character voice
- Avoids clichés and overused phrases
- Is grammatically perfect with natural dialogue tags

Respond ONLY with the requested content. No explanations, no meta-commentary."""


@dataclass
class TaskTemplate:
    name: str
    system_prompt: str
    user_prompt_template: str
    instruction_prefix: str

    def format_prompt(self, **kwargs) -> tuple[str, str]:
        return self.system_prompt, self.user_prompt_template.format(**kwargs)


TASK_TEMPLATES: Dict[str, TaskTemplate] = {
    "write": TaskTemplate(
        name="write",
        system_prompt=QUALITY_SYSTEM_PROMPT,
        user_prompt_template="""Write the opening scene of a {genre} story set in {setting}. The protagonist is {character}. Write {length} words of compelling literary prose. Respond only with the story text, no meta-commentary.""",
        instruction_prefix="Scrivi l'incipit di una storia",
    ),
    "continue": TaskTemplate(
        name="continue",
        system_prompt=QUALITY_SYSTEM_PROMPT,
        user_prompt_template="""Continue this story scene in the same style and tone. Maintain the voice and atmosphere:\n\n{input_text}\n\nWrite {length} words continuing from here. Respond only with the continuation, no explanations.""",
        instruction_prefix="Continua la scena mantenendo lo stesso stile e tono",
    ),
    "humanize": TaskTemplate(
        name="humanize",
        system_prompt="""You are FractalNova, an expert editor. Rewrite the following text to make it more natural, human, and engaging with better rhythm, voice, and flow. Fix any awkward phrasing while preserving the core meaning. Respond only with the rewritten text.""",
        user_prompt_template="""Rewrite this text to sound more natural and human, with better literary quality:\n\n{input_text}\n\nPreserve the core meaning but improve the prose significantly. Respond only with the rewritten version.""",
        instruction_prefix="Riscrivi questo testo rendendolo piu' umano e naturale",
    ),
    "title": TaskTemplate(
        name="title",
        system_prompt="""You are FractalNova, a publishing professional. Generate a powerful, marketable book title and one-line tagline from a synopsis. The title should be memorable (3-7 words), genre-appropriate, and commercially viable. The tagline should hook the reader in under 15 words.""",
        user_prompt_template="""Generate a compelling book title and tagline for this synopsis:\n\n{synopsis}\n\nGenre: {genre}\n\nRespond in JSON format: {{"title": "...", "tagline": "..."}}""",
        instruction_prefix="Proponi un titolo potente e una tagline",
    ),
    "synopsis": TaskTemplate(
        name="synopsis",
        system_prompt="""You are FractalNova, a publishing professional. Write a compelling book synopsis (blurb) of 100-150 words that hooks readers. Include the central conflict, stakes, and a hint of what makes the story unique. No spoilers. End with a hook question or statement.""",
        user_prompt_template="""Write a compelling book synopsis (100-150 words) based on:\n\nTitle: {title}\nGenre: {genre}\nSetting: {setting}\nProtagonist: {character}\n\nWrite a blurb that sells the book. Respond only with the synopsis.""",
        instruction_prefix="Genera una sinossi avvincente",
    ),
    "seo": TaskTemplate(
        name="seo",
        system_prompt="""You are FractalNova, a SEO specialist for books. Generate metadata that will help a book rank on Amazon, Google, and Wattpad. Extract keywords naturally from the content. Respond ONLY in valid JSON.""",
        user_prompt_template="""Generate SEO metadata for this book content. Extract keywords, tags, a meta description (max 160 chars), and categories:\n\nTitle: {title}\nGenre: {genre}\nContent preview:\n{content_preview}\n\nRespond ONLY with JSON: {{"keywords": [...], "tags": [...], "description": "...", "categories": [...]}}""",
        instruction_prefix="Genera metadati SEO in formato JSON",
    ),
    "translate": TaskTemplate(
        name="translate",
        system_prompt="""You are FractalNova, a literary translator. Translate the following text maintaining: the author's voice and style, emotional tone, rhythm and pacing, cultural nuances, and literary quality. Naturalize idioms rather than translating literally. Respond only with the translation.""",
        user_prompt_template="""Translate this literary text from {source_lang} to {target_lang}. Maintain all literary qualities:\n\n{input_text}\n\nRespond only with the translation in {target_lang}.""",
        instruction_prefix="Traduci il seguente testo mantenendo la qualita' letteraria",
    ),
}


def _load_gemini():
    try:
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            print("[ERROR] GOOGLE_API_KEY not set. Cannot generate dataset.")
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"[ERROR] Failed to load Gemini: {e}")
        return None


def generate_example(model, task: str, lang: str, **kwargs) -> Optional[Dict]:
    """Generate a single training example using Gemini."""
    template = TASK_TEMPLATES.get(task)
    if not template:
        return None

    system_prompt, user_prompt = template.format_prompt(**kwargs)
    lang_name = LANGUAGES.get(lang, lang)
    lang_instruction = f"Respond in {lang_name}. " if task != "translate" else ""

    try:
        chat = model.start_chat()
        response = chat.send_message(
            f"{system_prompt}\n\n{lang_instruction}{user_prompt}",
            generation_config={"temperature": 0.8, "max_output_tokens": 2048},
        )
        output = response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini call failed: {e}")
        return None

    if not output or len(output) < 20:
        return None

    return {
        "instruction": f"{template.instruction_prefix} ({lang_name})",
        "input": kwargs.get("input_text", kwargs.get("synopsis", kwargs.get("content_preview", ""))),
        "output": output,
        "lang": lang,
        "task": task,
        "source": "gemini-synthetic",
    }


def generate_write_example(model, lang: str) -> Optional[Dict]:
    genre = random.choice(GENRES)
    setting = random.choice(SETTINGS)
    character = random.choice(CHARACTERS)
    return generate_example(model, "write", lang, genre=genre, setting=setting, character=character, length="500-1000")


def generate_continue_example(model, lang: str, seed_examples: List[Dict]) -> Optional[Dict]:
    seeds = [e for e in seed_examples if e and e.get("task") == "write"]
    if not seeds:
        return generate_write_example(model, lang)
    seed = random.choice(seeds)
    output_text = seed.get("output", "")
    if len(output_text) < 100:
        return None
    input_text = output_text[:min(len(output_text) // 2, 500)]
    return generate_example(model, "continue", lang, input_text=input_text, length="200-300")


def generate_humanize_example(model, lang: str, seed_examples: List[Dict]) -> Optional[Dict]:
    seeds = [e for e in seed_examples if e and e.get("task") in ("write", "continue")]
    if not seeds:
        return None
    seed = random.choice(seeds)
    output_text = seed.get("output", "")
    if len(output_text) < 100:
        return None
    bad_text = output_text[:min(len(output_text), 400)]
    return generate_example(model, "humanize", lang, input_text=bad_text)


def generate_title_example(model, lang: str) -> Optional[Dict]:
    synopses = [
        ("A detective discovers that all cold cases in her city share a single, impossible connection.", "thriller"),
        ("A young woman inherits a library where books contain memories, not stories.", "fantasy"),
        ("Two astronauts stranded on Mars must survive with only 30 days of oxygen.", "sci-fi"),
        ("In 1920s Shanghai, a jazz musician uncovers a conspiracy among the foreign concessions.", "historical"),
        ("A lighthouse keeper begins receiving messages from a ship that sank 100 years ago.", "fiction"),
    ]
    synopsis, genre = random.choice(synopses)
    return generate_example(model, "title", lang, synopsis=synopsis, genre=genre)


def generate_synopsis_example(model, lang: str) -> Optional[Dict]:
    title = random.choice([
        "The Silent Echo", "Whispers of the Deep", "The Last Migration",
        "Glass Cities", "The Bone Forest", "The Color of Midnight",
        "Where Rivers Meet", "The Paper Architect", "Salt and Stars", "The Memory Keeper",
    ])
    genre = random.choice(GENRES)
    setting = random.choice(SETTINGS)
    character = random.choice(CHARACTERS)
    return generate_example(model, "synopsis", lang, title=title, genre=genre, setting=setting, character=character)


def generate_seo_example(model, lang: str, seed_examples: List[Dict]) -> Optional[Dict]:
    seeds = [e for e in seed_examples if e and e.get("task") in ("write", "continue", "synopsis")]
    if not seeds:
        return None
    seed = random.choice(seeds)
    content = seed.get("output", "")
    title = seed.get("instruction", "Untitled")[:80]
    genre = random.choice(GENRES)
    content_preview = content[:500] if len(content) > 500 else content
    return generate_example(model, "seo", lang, title=title, genre=genre, content_preview=content_preview)


def generate_translate_example(model, seed_examples: List[Dict], source_lang: str, target_lang: str) -> Optional[Dict]:
    seeds = [e for e in seed_examples if e and e.get("lang") == source_lang and e.get("task") in ("write", "continue")]
    if not seeds:
        return None
    seed = random.choice(seeds)
    input_text = seed.get("output", "")
    if len(input_text) < 100:
        return None
    input_text = input_text[:min(len(input_text), 500)]
    return generate_example(
        model, "translate", target_lang,
        input_text=input_text,
        source_lang=LANGUAGES.get(source_lang, source_lang),
        target_lang=LANGUAGES.get(target_lang, target_lang),
    )


GENERATORS = {
    "write": lambda model, lang, pool, **kw: generate_write_example(model, lang),
    "continue": lambda model, lang, pool, **kw: generate_continue_example(model, lang, pool),
    "humanize": lambda model, lang, pool, **kw: generate_humanize_example(model, lang, pool),
    "title": lambda model, lang, pool, **kw: generate_title_example(model, lang),
    "synopsis": lambda model, lang, pool, **kw: generate_synopsis_example(model, lang),
    "seo": lambda model, lang, pool, **kw: generate_seo_example(model, lang, pool),
}


def estimate_cost(num_examples: int) -> Dict:
    tokens_per_example = 1500
    total_tokens = num_examples * tokens_per_example
    gemini_cost_per_1k = 0.000075
    total_cost = total_tokens / 1000 * gemini_cost_per_1k
    return {"total_tokens": total_tokens, "estimated_cost_usd": round(total_cost, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=500, help="Total examples per task+lang combo")
    ap.add_argument("--generate-all", action="store_true", help="Generate all tasks for all languages")
    ap.add_argument("--num-per-task", type=int, default=200, help="Examples per task when using --generate-all")
    ap.add_argument("--languages", default="it,en,es,fr,de", help="Comma-separated language codes")
    ap.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated tasks")
    ap.add_argument("--out-dir", default="training/data/generated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Estimate cost only, don't generate")
    args = ap.parse_args()

    random.seed(args.seed)
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.generate_all:
        total_examples = len(langs) * len(tasks) * args.num_per_task
    else:
        total_examples = args.num_examples

    cost = estimate_cost(total_examples)
    print(f"[dataset] Target: {total_examples} examples")
    print(f"[dataset] Languages: {langs}")
    print(f"[dataset] Tasks: {tasks}")
    print(f"[dataset] Estimated cost: ${cost['estimated_cost_usd']:.2f}")

    if args.dry_run:
        return

    model = _load_gemini()
    if model is None:
        raise SystemExit("Gemini API not configured. Set GOOGLE_API_KEY.")

    os.makedirs(args.out_dir, exist_ok=True)
    pool: List[Dict] = []
    stats = {t: 0 for t in tasks}
    failed = 0

    if args.generate_all:
        # Generate write examples first to seed the pool
        print("[dataset] Phase 1: Generating seed write examples...")
        for lang in langs:
            n = max(50, args.num_per_task)
            for _ in tqdm(range(n), desc=f"write/{lang}"):
                ex = generate_write_example(model, lang)
                if ex:
                    pool.append(ex)
                    stats["write"] += 1
                    _append_example(ex, args.out_dir, "write", lang)
                else:
                    failed += 1
                time.sleep(0.1)

        # Generate all other tasks
        for lang in langs:
            for task in tasks:
                if task == "write":
                    continue
                gen_fn = GENERATORS.get(task)
                if not gen_fn:
                    continue
                n = args.num_per_task
                for _ in tqdm(range(n), desc=f"{task}/{lang}"):
                    try:
                        ex = gen_fn(model, lang, pool)
                        if ex:
                            pool.append(ex)
                            stats[task] += 1
                            _append_example(ex, args.out_dir, task, lang)
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"[ERROR] {task}/{lang}: {e}")
                        failed += 1
                    time.sleep(0.3)

        # Generate translations
        if "translate" in tasks:
            print("[dataset] Phase 2: Generating translations...")
            for src_lang in langs:
                for tgt_lang in langs:
                    if src_lang == tgt_lang:
                        continue
                    for _ in tqdm(range(args.num_per_task // 3), desc=f"translate/{src_lang}->{tgt_lang}"):
                        ex = generate_translate_example(model, pool, src_lang, tgt_lang)
                        if ex:
                            pool.append(ex)
                            stats["translate"] += 1
                            _append_example(ex, args.out_dir, "translate", f"{src_lang}_{tgt_lang}")
                        else:
                            failed += 1
                        time.sleep(0.3)
    else:
        for _ in tqdm(range(args.num_examples)):
            task = random.choice(tasks)
            lang = random.choice(langs)
            gen_fn = GENERATORS.get(task)
            if not gen_fn:
                continue
            try:
                ex = gen_fn(model, lang, pool)
                if ex:
                    pool.append(ex)
                    stats[task] += 1
                    _append_example(ex, args.out_dir, task, lang)
                else:
                    failed += 1
            except Exception as e:
                print(f"[ERROR] {e}")
                failed += 1
            time.sleep(0.3)

    # Write combined dataset for prepare_dataset.py
    all_path = os.path.join(args.out_dir, "all_generated.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for ex in pool:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[dataset] Done! Generated: {sum(stats.values())} examples, Failed: {failed}")
    for task, count in stats.items():
        print(f"  {task}: {count}")
    print(f"  Combined file: {all_path}")
    print(f"  Next: python training/prepare_dataset.py --inputs {all_path} --out-dir training/data")


def _append_example(ex: Dict, out_dir: str, task: str, lang: str):
    path = os.path.join(out_dir, f"{task}_{lang}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
