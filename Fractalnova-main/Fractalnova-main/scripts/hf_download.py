"""Downloader robusto da Hugging Face Hub.

Usa lo store certificati di Windows via `truststore` (risolve gli errori
CERTIFICATE_VERIFY_FAILED causati da antivirus/proxy che intercettano l'HTTPS).

Uso:
    python scripts/hf_download.py "Qwen/Qwen3-4B=models/Qwen3-4B" \
                                  "HuggingFaceTB/SmolLM2-1.7B-Instruct=models/SmolLM2-1.7B-Instruct"
"""
import sys

try:
    import truststore
    truststore.inject_into_ssl()
    print("[hf] truststore attivo (store certificati di Windows)")
except Exception as e:  # noqa: BLE001
    print(f"[hf] truststore non disponibile ({e}); uso certifi di default")

from huggingface_hub import snapshot_download  # noqa: E402

# Evita doppioni pesanti: formati alternativi (gguf/pth), pesi originali Mistral
# (consolidated.safetensors + original/) che duplicano i safetensors HF.
IGNORE = ["*.gguf", "*.pth", "*.pt", "consolidated*", "original/*", "*.msgpack", "*.h5"]


def main():
    pairs = sys.argv[1:]
    if not pairs:
        print("Specifica almeno un 'repo=cartella'")
        sys.exit(1)
    for pair in pairs:
        repo, _, out = pair.partition("=")
        out = out or f"models/{repo.split('/')[-1]}"
        print(f"[hf] download {repo} -> {out} (ignore: {IGNORE}) ...")
        path = snapshot_download(repo, local_dir=out, ignore_patterns=IGNORE)
        print(f"[hf] OK {repo} -> {path}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
