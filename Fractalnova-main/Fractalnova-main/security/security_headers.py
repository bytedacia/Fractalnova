"""
Pipeline Fase 2 - Hardening: Content Security Policy e headers di sicurezza.
Utilizzabile in Gradio (head=) o in un reverse proxy (nginx/Apache).
"""
from typing import Dict, Optional

# Content Security Policy - restrittiva per ridurre XSS e data injection
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Gradio richiede unsafe-inline/eval
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' https:; "
    "frame-ancestors 'self'; "
    "base-uri 'self'; "
    "form-action 'self';"
)

# Headers di sicurezza standard (OWASP)
SECURITY_HEADERS: Dict[str, str] = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Content-Security-Policy": DEFAULT_CSP,
}


def get_security_headers(csp: Optional[str] = None) -> Dict[str, str]:
    """Restituisce dizionario headers da applicare a ogni risposta."""
    h = dict(SECURITY_HEADERS)
    if csp is not None:
        h["Content-Security-Policy"] = csp
    return h


def gradio_head_html() -> str:
    """Meta tag e header-equivalenti per Gradio (injectati in head)."""
    parts = [
        '<meta http-equiv="X-Content-Type-Options" content="nosniff">',
        '<meta http-equiv="X-Frame-Options" content="SAMEORIGIN">',
        '<meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">',
    ]
    return "\n".join(parts)


def get_csp_for_proxy() -> str:
    """CSP da configurare in nginx/Apache/CloudFlare."""
    return SECURITY_HEADERS["Content-Security-Policy"]
