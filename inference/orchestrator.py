import os
from typing import Dict
from huggingface_hub import snapshot_download

from .generate import (
    generate_long_book,
    write_natural_chapter,
    qwen_humanize_and_proof,
    save_as_word,
    analyze_seo_with_gemma,
    build_professional_pitch,
    outreach_publishers,
    wattpad_export,
    llama_generate_text,
    generate_cover_with_flux,
)


class FractalNova:
    """
    Orchestratore end-to-end che combina generazione libro, raffinamento, SEO,
    outreach editori, export Wattpad e generazione copertina.
    """

    def __init__(self) -> None:
        self.author_name = os.getenv("AUTHOR_NAME", "")
        self.author_email = os.getenv("AUTHOR_EMAIL", "")
        self._ensure_models()

    def _ensure_models(self) -> None:
        models_map = {
            "QWEN_LOCAL_MODEL_PATH": os.getenv("QWEN_LOCAL_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Qwen3-8B")),
            "LLAMA_LOCAL_MODEL_PATH": os.getenv("LLAMA_LOCAL_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Meta-Llama-3-8B-Instruct")),
            "GEMMA_LOCAL_MODEL_PATH": os.getenv("GEMMA_LOCAL_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Gemma-7B")),
            "DEEPSEEK_LOCAL_PATH": os.getenv("DEEPSEEK_LOCAL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "DeepSeek-V3")),
            # FLUX is pulled by diffusers at runtime; optionally prefetch weights locally
            "FLUX_LOCAL_MODEL_PATH": os.getenv("FLUX_LOCAL_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "FLUX.1-schnell")),
        }
        repos = {
            "QWEN_LOCAL_MODEL_PATH": os.getenv("QWEN_REPO_ID", "Qwen/Qwen3-8B"),
            "LLAMA_LOCAL_MODEL_PATH": os.getenv("LLAMA_REPO_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
            "GEMMA_LOCAL_MODEL_PATH": os.getenv("GEMMA_REPO_ID", "google/gemma-7b-it"),
            "DEEPSEEK_LOCAL_PATH": os.getenv("DEEPSEEK_REPO_ID", "deepseek-ai/DeepSeek-V3"),
            "FLUX_LOCAL_MODEL_PATH": os.getenv("FLUX_REPO_ID", os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")),
        }
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        revision_overrides = {
            # Optional: allow pinning specific revisions
            "QWEN_LOCAL_MODEL_PATH": os.getenv("QWEN_REVISION"),
            "LLAMA_LOCAL_MODEL_PATH": os.getenv("LLAMA_REVISION"),
            "GEMMA_LOCAL_MODEL_PATH": os.getenv("GEMMA_REVISION"),
            "DEEPSEEK_LOCAL_PATH": os.getenv("DEEPSEEK_REVISION"),
            "FLUX_LOCAL_MODEL_PATH": os.getenv("FLUX_REVISION"),
        }
        for env_key, local_dir in models_map.items():
            if not os.path.exists(local_dir) or not os.listdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)
                repo_id = repos[env_key]
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    revision=revision_overrides.get(env_key) or None,
                )
            os.environ[env_key] = local_dir

    def run(self, book_details: Dict) -> Dict:
        # 1) struttura libro
        book_structure = generate_long_book(book_details)

        # 2) generazione e umanizzazione capitoli
        for chapter in book_structure['chapters']:
            style_guide = {"vocabulary": [], "sentence_structure": [], "themes": [], "techniques": []}
            chapter_content = write_natural_chapter(chapter, style_guide)
            chapter['content'] = qwen_humanize_and_proof(chapter_content)

        # 3) passaggio finale su tutto il libro
        full_text = []
        for chapter in book_structure['chapters']:
            full_text.append(f"# {chapter['title']}\n\n{chapter['content']}")
        refined_book = qwen_humanize_and_proof("\n\n".join(full_text))

        # 4) titolo e trama con Llama3 + umanizzazione Qwen
        llama_title_prompt = (
            "Leggi il seguente libro completo e proponi un titolo potente e sintetico (max 12 parole), "
            "in italiano, coerente con genere e tono. Restituisci solo il titolo.\n\n" + refined_book
        )
        llama_plot_prompt = (
            "Leggi il seguente libro completo e genera una sinossi/trama avvincente tra 120 e 200 parole, "
            "in italiano, senza spoiler e con focus su conflitto e temi.\n\n" + refined_book
        )
        raw_title = llama_generate_text(llama_title_prompt, temperature=0.5, max_new_tokens=64)
        raw_plot = llama_generate_text(llama_plot_prompt, temperature=0.5, max_new_tokens=220)
        human_title = qwen_humanize_and_proof(raw_title, temperature=0.6, max_new_tokens=128).strip()
        human_plot = qwen_humanize_and_proof(raw_plot, temperature=0.6, max_new_tokens=512).strip()
        if human_title:
            book_structure['title'] = human_title
        if human_plot:
            book_structure['plot'] = human_plot

        # 5) SEO Gemma
        seo = analyze_seo_with_gemma(refined_book)
        book_structure['seo'] = seo

        # 6) export Wattpad txt e Word
        wattpad_path = wattpad_export(book_structure)
        book_structure['wattpad_path'] = wattpad_path
        word_path = save_as_word(book_structure, book_structure['title'])

        # 7) copertina con Flux
        cover_path = generate_cover_with_flux(book_structure.get('title', ''), refined_book, book_structure.get('genre', ''))
        book_structure['cover_path'] = cover_path

        # 8) outreach editori via email
        pitch = build_professional_pitch(book_structure.get('title', ''), book_structure.get('plot', ''), seo)
        outreach = outreach_publishers(
            book_structure.get('title', ''),
            pitch,
            [
                "casa editrice narrativa contatti email",
                "editori italiani invio manoscritti",
                "publishers fiction submissions email",
            ],
            attachments=[p for p in [word_path, wattpad_path] if p]
        )
        book_structure['outreach'] = outreach

        return {
            "book_structure": book_structure,
            "word_path": word_path,
            "wattpad_path": wattpad_path,
            "cover_path": cover_path,
            "seo": seo,
            "outreach": outreach,
        }


