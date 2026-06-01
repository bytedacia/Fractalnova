"""Publishing: pacchetto di sottomissione, metadati KDP e outreach con consenso.

Principio: **nessuna email viene inviata automaticamente.** L'outreach produce un
PIANO da approvare manualmente (evita spam e problemi legali).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

from . import seo as seo_mod
from .logging_config import get_logger

log = get_logger(__name__)

KDP_MAX_KEYWORDS = 7
KDP_MAX_CATEGORIES = 2


def build_kdp_metadata(book: Dict, seo_data=None, author: Optional[str] = None, language: str = "it") -> Dict:
    seo_norm = seo_mod.parse_seo(seo_data if seo_data is not None else book.get("seo", {}))
    return {
        "title": book.get("title", ""),
        "subtitle": book.get("subtitle", ""),
        "author": author or book.get("author") or os.getenv("AUTHOR_NAME", ""),
        "language": language,
        "description": seo_norm["description"] or book.get("plot", ""),
        "keywords": seo_norm["keywords"][:KDP_MAX_KEYWORDS],   # KDP: max 7 keyword
        "categories": seo_norm["categories"][:KDP_MAX_CATEGORIES],  # KDP: max 2 categorie
    }


def _safe(title: str) -> str:
    name = "".join(c for c in (title or "libro") if c.isalnum() or c in (" ", "-", "_")).strip()
    return (name or "libro").replace(" ", "_")[:80]


def build_submission_package(
    book: Dict,
    assets: Optional[Dict[str, Optional[str]]] = None,
    out_dir: str = "exports/submissions",
    author: Optional[str] = None,
) -> str:
    """Crea una cartella pronta per l'invio: manoscritti, copertina, sinossi, metadati."""
    folder = os.path.join(out_dir, _safe(book.get("title")))
    os.makedirs(folder, exist_ok=True)

    copied: Dict[str, str] = {}
    for kind, path in (assets or {}).items():
        if path and os.path.exists(path):
            dst = os.path.join(folder, os.path.basename(path))
            shutil.copy2(path, dst)
            copied[kind] = os.path.basename(dst)

    with open(os.path.join(folder, "synopsis.txt"), "w", encoding="utf-8") as f:
        f.write(book.get("plot", ""))

    metadata = build_kdp_metadata(book, author=author)
    metadata["files"] = copied
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info("Pacchetto di sottomissione creato: %s", folder)
    return folder


def plan_outreach(recipients: List[str], book_title: str, pitch: str) -> Dict:
    """Restituisce un PIANO di invio (NON invia). L'invio resta una decisione umana."""
    unique = list(dict.fromkeys(r.strip() for r in recipients if r and r.strip()))
    return {
        "book_title": book_title,
        "pitch_preview": (pitch or "")[:300],
        "recipients": unique,
        "count": len(unique),
        "status": "DA_APPROVARE",
        "note": "Nessuna email inviata. Rivedere e approvare manualmente prima dell'invio.",
    }
