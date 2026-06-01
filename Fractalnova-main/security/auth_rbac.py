"""
Pipeline Fase 3 - Autenticazione e autorizzazione: policy password, RBAC, JWT, MFA stub.
"""
import os
import re
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from functools import wraps

# Ruoli e permessi (POLP)
ROLES = {
    "admin": {"users", "config", "models", "security", "backup", "audit"},
    "editor": {"generate", "export", "edit_books"},
    "viewer": {"read", "export"},
}

# Password policy
MIN_LENGTH = 12
MAX_LENGTH = 128
REQUIRE_UPPER = True
REQUIRE_LOWER = True
REQUIRE_DIGIT = True
REQUIRE_SPECIAL = True
SPECIAL_CHARS = r"!@#$%^&*()_+-=[]{}|;:',.<>?/"


def validate_password_strength(password: str) -> tuple:
    """
    Restituisce (ok: bool, errors: list).
    """
    errors = []
    if len(password) < MIN_LENGTH:
        errors.append(f"Lunghezza minima {MIN_LENGTH} caratteri")
    if len(password) > MAX_LENGTH:
        errors.append(f"Lunghezza massima {MAX_LENGTH} caratteri")
    if REQUIRE_UPPER and not re.search(r"[A-Z]", password):
        errors.append("Almeno un carattere maiuscolo")
    if REQUIRE_LOWER and not re.search(r"[a-z]", password):
        errors.append("Almeno un carattere minuscolo")
    if REQUIRE_DIGIT and not re.search(r"\d", password):
        errors.append("Almeno una cifra")
    if REQUIRE_SPECIAL and not re.search(r"[!@#$%^&*()_+\-=[\]{}|;:',.<>?/]", password):
        errors.append("Almeno un carattere speciale")
    return (len(errors) == 0, errors)


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple:
    """Hash con salt (compatibile con verifiche successive). Restituisce (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return key.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Verifica password rispetto a hash e salt memorizzati."""
    try:
        salt = bytes.fromhex(stored_salt)
        key, _ = hash_password(password, salt)
        return hmac.compare_digest(key, stored_hash)
    except Exception:
        return False


def create_jwt_payload(user_id: str, role: str, extra: Optional[Dict] = None) -> dict:
    """Crea payload JWT (firma effettiva richiede PyJWT o simile). Qui solo struttura."""
    now = datetime.utcnow()
    return {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "nbf": now,
        **(extra or {}),
    }


def has_permission(role: str, permission: str) -> bool:
    """Verifica se il ruolo ha il permesso (RBAC)."""
    perms = ROLES.get(role) or set()
    return permission in perms


def get_roles() -> Dict[str, Set[str]]:
    """Restituisce matrice ruoli/permessi."""
    return {k: set(v) for k, v in ROLES.items()}


# MFA stub: in produzione usare TOTP (pyotp) o WebAuthn
def mfa_required_for_admin() -> bool:
    """Se MFA è obbligatoria per admin (configurabile)."""
    return os.getenv("MFA_REQUIRED_ADMIN", "true").lower() in ("1", "true", "yes")


def check_api_key(provided: Optional[str]) -> bool:
    """Verifica header X-API-KEY contro API_SECRET."""
    secret = os.getenv("API_SECRET")
    if not secret:
        return True  # Nessuna auth configurata
    return provided is not None and hmac.compare_digest(secret, provided)


def require_permission(permission: str):
    """Decorator stub: in app reale estrarre ruolo da JWT/sessione e chiamare has_permission."""
    def deco(f):
        @wraps(f)
        def inner(*args, **kwargs):
            # Qui andrebbe ruolo da request/context
            return f(*args, **kwargs)
        return inner
    return deco
