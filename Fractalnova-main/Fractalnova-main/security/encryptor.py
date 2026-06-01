import os
from typing import Optional
from cryptography.fernet import Fernet


def _get_or_create_key() -> bytes:
    key = os.getenv("SECURITY_ENC_KEY")
    if key:
        return key.encode("utf-8")
    key = Fernet.generate_key()
    os.environ["SECURITY_ENC_KEY"] = key.decode("utf-8")
    return key


def encrypt_file(target_path: str, output_path: Optional[str] = None) -> Optional[str]:
    if not os.path.isfile(target_path):
        return None
    key = _get_or_create_key()
    f = Fernet(key)
    with open(target_path, "rb") as rf:
        data = rf.read()
    token = f.encrypt(data)
    out = output_path or f"{target_path}.enc"
    with open(out, "wb") as wf:
        wf.write(token)
    return out


def decrypt_file(enc_path: str, output_path: Optional[str] = None) -> Optional[str]:
    if not os.path.isfile(enc_path):
        return None
    key = _get_or_create_key()
    f = Fernet(key)
    with open(enc_path, "rb") as rf:
        token = rf.read()
    data = f.decrypt(token)
    out = output_path or enc_path.replace(".enc", "")
    with open(out, "wb") as wf:
        wf.write(data)
    return out


