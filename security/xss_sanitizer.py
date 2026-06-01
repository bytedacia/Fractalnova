"""
Pipeline Fase 5 - Sicurezza codice: escape output per prevenire XSS.
"""
import html
import re
from typing import Optional


def escape_html(text: Optional[str]) -> str:
    """Escape caratteri HTML per evitare XSS in output."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def strip_tags(html_fragment: str, allowed: Optional[set] = None) -> str:
    """Rimuove tag HTML non consentiti. allowed es. {'b','i','a'}."""
    if not html_fragment:
        return ""
    if allowed is None:
        allowed = set()
    # Rimuovi tutti i tag
    out = re.sub(r"<[^>]+>", "", html_fragment)
    return escape_html(out)


def safe_attr_value(value: Optional[str]) -> str:
    """Valore sicuro per attributi HTML (escape + quote)."""
    if value is None:
        return ""
    return escape_html(str(value)).replace('"', "&quot;")
