"""Generazione di un libro completo tramite il tier modello (multi-step).

Outline -> capitoli -> titolo/sinossi -> SEO. Restituisce una struttura
compatibile con `fractalnova.export` e con la pipeline esistente.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from fractalnova.logging_config import get_logger
from fractalnova.seo import parse_seo

from .client import chat

log = get_logger("fractalnova.serving")

SYSTEM = (
    "Sei FractalNova, autore ed editor professionista. Scrivi in modo naturale e umano, "
    "con voce e ritmo curati, rispondendo SEMPRE nella lingua della richiesta."
)


def _json_array(text: str) -> List[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [ln.strip("-•* \t") for ln in text.splitlines() if ln.strip()]


def generate_book(details: Dict[str, Any]) -> Dict[str, Any]:
    lang = details.get("language", "it")
    genre = details.get("genre", "fiction")
    base_title = details.get("title") or ""
    pages = int(details.get("target_pages", 100) or 100)
    n_chapters = max(1, min(pages // 25, 200))  # nessun limite pratico

    # 1) Outline
    titles = _json_array(chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": (
             f"Proponi {n_chapters} titoli di capitolo (lingua: {lang}) per un libro "
             f"'{base_title or 'senza titolo'}' di genere {genre}. "
             "Rispondi SOLO con un array JSON di stringhe.")}],
        temperature=0.7, max_tokens=8192,
    ))[:n_chapters]
    if not titles:
        titles = [f"Capitolo {i}" for i in range(1, n_chapters + 1)]

    # 2) Capitoli
    chapters = []
    for title in titles:
        content = chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": (
                 f"Scrivi il capitolo intitolato '{title}' (lingua {lang}, genere {genre}), "
                 "ricco e coerente, senza limiti di lunghezza. Restituisci solo il testo del capitolo.")}],
            temperature=0.9, max_tokens=32768,
        )
        chapters.append({"title": title, "content": content.strip()})

    # 3) Titolo, sinossi, SEO
    preview = "\n\n".join(c["content"][:500] for c in chapters[:3])
    title = base_title or chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Proponi un titolo potente (lingua {lang}). Solo il titolo.\n\n{preview}"}],
        temperature=0.6, max_tokens=256,
    ).strip()
    plot = chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Scrivi una sinossi avvincente (max 120 parole, lingua {lang}).\n\n{preview}"}],
        temperature=0.6, max_tokens=2048,
    ).strip()
    seo = parse_seo(chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": (
             f"Genera metadati SEO in JSON (keywords, tags, description, categories), lingua {lang}.\n\n{plot}")}],
        temperature=0.3, max_tokens=4096,
    ))

    return {
        "title": title, "genre": genre, "plot": plot, "language": lang,
        "seo": seo, "chapters": chapters, "_mode": "model-server",
    }
