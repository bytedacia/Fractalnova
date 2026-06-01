"""Estrazione e validazione robusta dei metadati SEO prodotti da un LLM.

I modelli spesso restituiscono JSON "sporco" (con testo intorno, virgolette
tipografiche, ecc.). Qui lo estraiamo e normalizziamo in uno schema stabile.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from .logging_config import get_logger

log = get_logger(__name__)

MAX_DESCRIPTION = 160


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _loads_lenient(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # estrai il primo blocco {...} bilanciato in modo best-effort
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0).replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            log.warning("SEO: blocco JSON trovato ma non parsabile")
    return None


def normalize_seo(data: Dict[str, Any]) -> Dict[str, Any]:
    description = str(data.get("description", "")).strip()
    if len(description) > MAX_DESCRIPTION:
        description = description[: MAX_DESCRIPTION - 3].rstrip() + "..."
    return {
        "keywords": _to_list(data.get("keywords"))[:20],
        "tags": _to_list(data.get("tags"))[:20],
        "description": description,
        "categories": _to_list(data.get("categories"))[:10],
    }


def parse_seo(text_or_data: Any) -> Dict[str, Any]:
    """Accetta una stringa (output del modello) o un dict gia' pronto."""
    if isinstance(text_or_data, dict):
        return normalize_seo(text_or_data)
    data = _loads_lenient(str(text_or_data)) or {}
    if not data:
        log.info("SEO: nessun JSON valido, restituisco schema vuoto")
    return normalize_seo(data)
