"""Logica di business: esecuzione dei job di generazione libro.

`process_job` gira in background. Tenta la pipeline reale (orchestratore con
DeepSeek/Qwen/Llama/Gemma/FLUX); se i modelli non sono disponibili usa un
generatore segnaposto, cosi' l'API e i test funzionano ovunque.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fractalnova.export import export_all
from fractalnova.logging_config import get_logger

from .db import Book, Job, SessionLocal

log = get_logger("fractalnova.api")


def _stub_book(details: Dict[str, Any]) -> Dict[str, Any]:
    title = details.get("title") or "Bozza FractalNova"
    genre = details.get("genre", "fiction")
    pages = int(details.get("target_pages", 100) or 100)
    n_chapters = max(1, min(pages // 25, 40))
    chapters = [
        {
            "title": f"Capitolo {i}",
            "content": (
                "Capitolo segnaposto generato in modalita' degradata (modelli non "
                "disponibili in questo ambiente). In produzione qui va il testo reale "
                f"prodotto da FractalNova-Pro.\n\nGenere: {genre}. Capitolo {i} di {n_chapters}."
            ),
        }
        for i in range(1, n_chapters + 1)
    ]
    return {
        "title": title,
        "genre": genre,
        "plot": details.get("plot", ""),
        "language": details.get("language", "it"),
        "seo": {
            "keywords": [genre],
            "tags": [],
            "description": (details.get("plot", "") or title)[:160],
            "categories": [genre],
        },
        "chapters": chapters,
        "_mode": "stub",
    }


def _generate_structure(details: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Tier modello dedicato (vLLM) se configurato -> generazione reale.
    try:
        from serving.client import is_configured
        if is_configured():
            from serving.generation import generate_book
            return generate_book(details)
    except Exception as exc:  # noqa: BLE001
        log.warning("Model tier non disponibile (%s)", exc)

    # 2) Orchestratore locale (pesi su questa macchina).
    try:
        from inference.orchestrator import FractalNova  # import pesante: solo se necessario
        result = FractalNova().run(details)
        return result.get("book_structure", result)
    except Exception as exc:  # noqa: BLE001
        log.warning("Pipeline locale non disponibile (%s) -> generatore segnaposto", exc)

    # 3) Segnaposto: l'API non crasha mai.
    return _stub_book(details)


def process_job(job_id: str, user_id: int, details: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        db.commit()

        structure = _generate_structure(details)
        formats = export_all(structure, cover_path=structure.get("cover_path"))

        book = Book(
            user_id=user_id,
            title=structure.get("title") or details.get("title") or "Senza Titolo",
            genre=details.get("genre", "fiction"),
            status="completed",
            content_json=json.dumps(structure, ensure_ascii=False),
            formats_json=json.dumps({k: v for k, v in formats.items() if v}, ensure_ascii=False),
        )
        db.add(book)
        db.commit()
        db.refresh(book)

        job.book_id = book.id
        job.status = "completed"
        db.commit()
        log.info("Job %s completato -> book %s", job_id, book.id)
    except Exception as exc:  # noqa: BLE001 — vogliamo registrare qualsiasi fallimento del job
        log.exception("Job %s fallito", job_id)
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error = str(exc)[:2000]
            db.commit()
    finally:
        db.close()
