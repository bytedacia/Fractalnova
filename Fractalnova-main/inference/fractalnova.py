import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Path del modello fine-tunato FractalNova-Pro
FN_PRO_DIR = os.getenv("FRACTALNOVA_PRO_DIR", "training/outputs/fractalnova-pro")
FN_PRO_MANIFEST = os.path.join(FN_PRO_DIR, "manifest.json")


@dataclass
class FractalNovaProfile:
    model_id: str = "Qwen/Qwen3-4B"
    adapter_path: Optional[str] = None
    vision_model: Optional[str] = None
    vision_adapter: Optional[str] = None
    cover_model: Optional[str] = None
    cover_adapter: Optional[str] = None
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    max_seq_len: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def _load_manifest() -> Optional[Dict]:
    """Carica manifest.json del fine-tune FractalNova-Pro."""
    if not os.path.exists(FN_PRO_MANIFEST):
        return None
    with open(FN_PRO_MANIFEST) as f:
        return json.load(f)


def _build_pro_profile_from_manifest() -> FractalNovaProfile:
    """Costruisce il profilo Pro dal manifest se presente, altrimenti default."""
    m = _load_manifest()
    if m and "models" in m:
        text = m["models"].get("text", {})
        vision = m["models"].get("vision", {})
        cover = m["models"].get("cover", {})
        return FractalNovaProfile(
            model_id=text.get("base", "Qwen/Qwen3-4B"),
            adapter_path=text.get("adapter"),
            vision_model=vision.get("base"),
            vision_adapter=vision.get("adapter"),
            cover_model=cover.get("base"),
            cover_adapter=cover.get("adapter"),
        )
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fractalnova-main/
    return FractalNovaProfile(
        model_id=os.getenv("FRACTALNOVA_PRO_MODEL", os.path.join(base, "models", "Qwen3-4B")),
        adapter_path=os.getenv("FRACTALNOVA_PRO_ADAPTER"),
        vision_model=os.getenv("GEMMA4_MODEL_ID", os.path.join(base, "models", "gemma-4-E2B")),
        vision_adapter=os.getenv("FRACTALNOVA_VISION_ADAPTER"),
        cover_model=os.getenv("FLUX_MODEL_ID", os.path.join(base, "models", "FLUX.1-dev")),
        cover_adapter=os.getenv("FRACTALNOVA_COVER_ADAPTER"),
        max_seq_len=int(os.getenv("FRACTALNOVA_MAX_SEQ_LEN", "8192")),
    )


CORE_PROFILE = FractalNovaProfile(
    model_id=os.getenv("FRACTALNOVA_CORE_MODEL", "models/fractalnova-core-124m"),
    max_seq_len=int(os.getenv("FRACTALNOVA_CORE_MAX_SEQ_LEN", "2048")),
    temperature=0.8,
)


class FractalNovaInference:
    """Motore unico FractalNova.

    Ogni chiamata carica lazy il modello giusto:
    - FractalNova-Pro (Qwen3-4B + Gemma-4-E2B + FLUX.1-dev, tutti fine-tunati)
    - FractalNova-Core (124M da zero)

    Se esiste training/outputs/fractalnova-pro/manifest.json,
    carica automaticamente tutti e 3 i fine-tune.
    """

    def __init__(self, profile: str = "pro"):
        if profile == "pro":
            self.profile = _build_pro_profile_from_manifest()
        elif profile == "core":
            self.profile = CORE_PROFILE
        else:
            raise ValueError(f"Profilo sconosciuto: {profile}. Usa 'pro' o 'core'.")

        self._model = None
        self._tokenizer = None
        self._flux_pipe = None
        self._vision_model = None
        self._vision_processor = None
        self._core_model = None
        self._core_args = None
        self._is_finetuned = _load_manifest() is not None

    @property
    def model_name(self) -> str:
        return "FractalNova-Pro (fine-tuned)" if self._is_finetuned else "FractalNova-Pro (base)"

    # ── lazy load ────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.profile.model_id)

        kwargs = {
            "torch_dtype": getattr(torch, self.profile.dtype),
            "device_map": "auto",
        }
        if self.profile.adapter_path and os.path.isdir(self.profile.adapter_path):
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(self.profile.model_id, **kwargs)
            self._model = PeftModel.from_pretrained(base, self.profile.adapter_path)
            print(f"[FractalNova] loaded fine-tuned text adapter: {self.profile.adapter_path}")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.profile.model_id, **kwargs)

    def _load_flux(self):
        if self._flux_pipe is not None:
            return
        import torch
        from diffusers import DiffusionPipeline

        model_id = self.profile.cover_model or os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
        self._flux_pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            self._flux_pipe = self._flux_pipe.to("cuda")

        adapter = self.profile.cover_adapter
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel
            self._flux_pipe.transformer = PeftModel.from_pretrained(
                self._flux_pipe.transformer, adapter
            )
            print(f"[FractalNova] loaded fine-tuned cover adapter: {adapter}")

    def _load_vision(self):
        if self._vision_model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = self.profile.vision_model or os.getenv("GEMMA4_MODEL_ID", "google/gemma-4-E2B")
        self._vision_processor = AutoProcessor.from_pretrained(model_id)
        self._vision_model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

        adapter = self.profile.vision_adapter
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel
            self._vision_model = PeftModel.from_pretrained(self._vision_model, adapter)
            print(f"[FractalNova] loaded fine-tuned vision adapter: {adapter}")

    # ── testo ────────────────────────────────────────────────────

    def generate(self, prompt: str, *, temperature: float = None, max_tokens: int = None, **kwargs) -> str:
        self._load()
        import torch
        temp = temperature if temperature is not None else self.profile.temperature
        mt = max_tokens if max_tokens is not None else self.profile.max_seq_len

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self.profile.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=mt, temperature=temp,
                top_p=self.profile.top_p, repetition_penalty=self.profile.repetition_penalty,
                do_sample=True,
            )
        return self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def humanize(self, text: str, *, temperature: float = 0.6) -> str:
        return self.generate(
            f"Riscrivi il seguente testo in modo naturale, scorrevole e umano, "
            f"mantenendo tutto il contenuto informativo:\n\n{text}",
            temperature=temperature, max_tokens=int(len(text) * 1.5) + 512,
        )

    def translate(self, text: str, target_lang: str = "en") -> str:
        lang_names = {"it": "italiano", "en": "inglese", "es": "spagnolo", "fr": "francese",
                      "de": "tedesco", "pt": "portoghese", "ru": "russo", "zh": "cinese",
                      "ja": "giapponese", "ar": "arabo"}
        return self.generate(
            f"Traduci il seguente testo in {lang_names.get(target_lang, target_lang)}, "
            f"mantenendo lo stile letterario:\n\n{text}", temperature=0.3,
        )

    def chat(self, messages: List[Dict], *, temperature: float = None, max_tokens: int = None) -> str:
        self._load()
        import torch
        temp = temperature or self.profile.temperature
        mt = max_tokens or 2048
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self.profile.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=mt, temperature=temp, top_p=self.profile.top_p, do_sample=True)
        return self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # ── analisi testo ────────────────────────────────────────────

    def analyze_style(self, text: str) -> Dict:
        result = self.generate(
            f"Analizza lo stile del seguente testo. Restituisci solo JSON con: "
            f"vocabulary_level, sentence_structure, tone, readability, suggestions (lista):\n\n{text}",
            temperature=0.2, max_tokens=1024,
        )
        try:
            start = result.index("{")
            return json.loads(result[start:result.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return {"vocabulary_level": "unknown", "tone": "unknown", "suggestions": []}

    def analyze_seo(self, book_text: str) -> Dict:
        raw = self.generate(
            f"Analizza SEO del seguente libro. Restituisci solo JSON con: "
            f"keywords (lista), description, title_tag:\n\n{book_text[:3000]}",
            temperature=0.2, max_tokens=512,
        )
        try:
            start = raw.index("{")
            return json.loads(raw[start:raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return {"keywords": [], "description": "", "title_tag": ""}

    # ── visione (Gemma-4-E2B fine-tunato) ─────────────────────────

    def _vision_infer(self, image_path: str, text_prompt: str) -> str:
        from PIL import Image
        self._load_vision()
        image = Image.open(image_path).convert("RGB")
        inputs = self._vision_processor(text=text_prompt, images=image, return_tensors="pt").to(self.profile.device)
        import torch
        with torch.no_grad():
            out = self._vision_model.generate(**inputs, max_new_tokens=1024)
        return self._vision_processor.decode(out[0], skip_special_tokens=True).strip()

    def describe_image(self, image_path: str) -> str:
        return self._vision_infer(
            image_path,
            "Describe this book cover image in detail: composition, colors, mood, typography, genre signals."
        )

    def analyze_image_seo(self, image_path: str) -> Dict:
        raw = self._vision_infer(
            image_path,
            "Analyze this book cover for SEO metadata. Return ONLY JSON with: "
            "alt_text, keywords (list of 10), description (max 160 chars), "
            "visual_style, suggested_title_variations (list of 3)."
        )
        try:
            start = raw.index("{")
            return json.loads(raw[start:raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return {"alt_text": "", "keywords": [], "description": "", "visual_style": "", "suggested_title_variations": []}

    def classify_genre_from_cover(self, image_path: str) -> str:
        return self._vision_infer(
            image_path,
            "What genre? Answer one word: fantasy, sci-fi, romance, thriller, horror, historical, literary, non-fiction, or unknown."
        )

    # ── copertina (FLUX.1-dev fine-tunato) ────────────────────────

    def generate_cover(self, title: str, genre: str = "fiction", *, style: str = "cinematografico",
                       theme: str = "", context: str = "") -> Optional[str]:
        try:
            self._load_flux()
        except Exception:
            return None
        prompt = (
            f"Book cover for '{title}'. Genre: {genre}. Style: {style}. {theme}. "
            f"{context}. Professional layout, evocative, readable typography, 8k."
        )
        image = self._flux_pipe(prompt=prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
        os.makedirs("book_covers", exist_ok=True)
        safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip() or "cover"
        path = os.path.join("book_covers", f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.png")
        image.save(path)
        return path

    def generate_cover_variants(self, title: str, genre: str = "fiction", n: int = 4) -> List[str]:
        styles = ["cinematografico", "minimalista", "acquerello", "vintage", "onirico", "grafico vettoriale"]
        paths = []
        for i in range(min(n, len(styles))):
            p = self.generate_cover(title, genre, style=styles[i], theme=f"variant {i+1}")
            if p:
                paths.append(p)
        return paths

    # ── libro ────────────────────────────────────────────────────

    def generate_book(self, details: Dict) -> Dict:
        # Scrive i capitoli con QUESTO modello (Pro/Core), non con la pipeline
        # DeepSeek: il libro nasce dal modello caricato qui.
        from inference.generate import generate_long_book
        book = generate_long_book(details)
        genre = details.get("genre", "")
        lang = details.get("language", "it")
        for ch in book.get("chapters", []):
            ch["content"] = self.generate(
                f"Scrivi il capitolo '{ch.get('title', '')}' (lingua {lang}, genere {genre}), "
                f"ricco e coerente. Restituisci solo il testo del capitolo.",
                temperature=0.9,
            )
        book["full_text"] = "\n\n".join(
            f"# {c['title']}\n\n{c['content']}" for c in book.get("chapters", [])
        )
        return book

    # ── pipeline ─────────────────────────────────────────────────

    def run(self, book_details: Dict) -> Dict:
        book = self.generate_book(book_details)
        seo = self.analyze_seo(book.get("full_text", ""))
        book["seo"] = seo
        cover = self.generate_cover(
            book_details.get("title", ""), book_details.get("genre", ""),
            context=book.get("full_text", "")[:500],
        )
        book["cover_path"] = cover
        return book

    def run_with_vision(self, book_details: Dict, cover_image: str = None) -> Dict:
        result = self.run(book_details)
        img = cover_image or result.get("cover_path")
        if img and os.path.exists(img):
            result["cover_seo"] = self.analyze_image_seo(img)
            result["cover_description"] = self.describe_image(img)
            result["detected_genre"] = self.classify_genre_from_cover(img)
        return result


# ── download manager ─────────────────────────────────────────────

MODEL_REGISTRY = {
    "pro": {"repo": "Qwen/Qwen3-4B", "path": "models/Qwen3-4B"},
    "flux": {"repo": "black-forest-labs/FLUX.1-dev", "path": "models/FLUX.1-dev"},
    "gemma4": {"repo": "google/gemma-4-E2B", "path": "models/gemma-4-E2B"},
    "core": {"path": "models/fractalnova-core-124m", "local": True},
}


def download_models(selected: List[str] = None, token: str = None):
    from huggingface_hub import snapshot_download
    selected = selected or list(MODEL_REGISTRY.keys())
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    for name in selected:
        info = MODEL_REGISTRY[name]
        if info.get("local"):
            print(f"[FractalNova]  {name}: locale, skip")
            continue
        dest = info["path"]
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"[FractalNova]  {name}: già presente in {dest}")
            continue
        print(f"[FractalNova]  download {info['repo']} → {dest} ...")
        os.makedirs(dest, exist_ok=True)
        try:
            snapshot_download(repo_id=info["repo"], local_dir=dest,
                              local_dir_use_symlinks=False, token=token)
            print(f"[FractalNova]  ✓ {name} completo")
        except Exception as e:
            print(f"[FractalNova]  ✗ {name} fallito: {e}")


def download_main():
    import argparse
    ap = argparse.ArgumentParser(description="FractalNova · download modelli")
    ap.add_argument("--download", nargs="*", choices=list(MODEL_REGISTRY),
                    default=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--token")
    args = ap.parse_args()
    download_models(args.download, args.token)


if __name__ == "__main__":
    download_main()
