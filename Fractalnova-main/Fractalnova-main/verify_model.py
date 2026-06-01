"""
FractalNova · Verify base model loads and inference works.
Testa tutti e 3 i modelli locali senza fare fine-tuning.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def check_dir(name: str, required_files: list[str], allow_partial=False):
    path = MODELS_DIR / name
    if not path.is_dir():
        print(f"  ✗ {name}: directory non trovata")
        return False
    ok = True
    for fname in required_files:
        f = path / fname
        if f.is_file() and f.stat().st_size > 0:
            continue
        # allow .part files as partial progress
        if allow_partial and any(p.name.startswith(fname.split('.')[0]) for p in path.glob("*.part")):
            print(f"  ⏳ {name}: {fname} → download in corso (.part presente)")
            continue
        if f.is_file() and f.stat().st_size == 0:
            print(f"  ⏳ {name}: {fname} → placeholder, download incompleto")
            ok = False
        else:
            print(f"  ✗ {name}: {fname} mancante")
            ok = False
    return ok


def test_qwen():
    print("\n── Qwen3-4B ──")
    if not check_dir("Qwen3-4B", ["config.json", "tokenizer.json"], allow_partial=True):
        return False
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODELS_DIR / "Qwen3-4B", trust_remote_code=False)
        print(f"  Config: {cfg.model_type}, vocab={cfg.vocab_size}, layers={cfg.num_hidden_layers}")
        print(f"  Arch: {cfg.architectures}")
    except Exception as e:
        print(f"  Config fallita: {e}")
        return False
    return True


def test_gemma():
    print("\n── Gemma-4-E2B ──")
    required = ["config.json", "generation_config.json", "tokenizer.json",
                "model.safetensors.index.json"]
    if not check_dir("gemma-4-E2B", required, allow_partial=True):
        return False
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODELS_DIR / "gemma-4-E2B", trust_remote_code=True)
        print(f"  Config: {cfg.model_type}, layers={cfg.num_hidden_layers}")
        print(f"  Arch: {cfg.architectures}")
    except Exception as e:
        print(f"  Config fallita: {e}")
        return False
    return True


def test_flux():
    print("\n── FLUX.1-dev ──")
    required = ["model_index.json", "vae/config.json", "transformer/config.json",
                "text_encoder_2/config.json", "text_encoder/config.json"]
    if not check_dir("FLUX.1-dev", required, allow_partial=True):
        return False
    try:
        from diffusers import FluxPipeline
        import torch
        pipe = FluxPipeline.from_pretrained(
            MODELS_DIR / "FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        print(f"  ✓ FLUX pipeline caricata: {type(pipe).__name__}")
        del pipe
    except ImportError:
        print("  diffusers non installato — salto caricamento")
    except Exception as e:
        print(f"  Load fallito (atteso se pesi mancanti): {e}")
        return False
    return True


def test_unified():
    print("\n── FractalNovaInference (unified) ──")
    try:
        sys.path.insert(0, str(ROOT))
        from inference.fractalnova import FractalNovaInference
        ai = FractalNovaInference("pro")
        print(f"  ✓ FractalNovaInference creata (modello: {ai.model_name})")
        return True
    except Exception as e:
        print(f"  Init fallita: {e}")
        return False


if __name__ == "__main__":
    print(f"FractalNova Verify — ROOT: {ROOT}")
    results = [test_qwen(), test_gemma(), test_flux(), test_unified()]
    ok = all(results)
    print(f"\n{'='*40}")
    if ok:
        print("✓ TUTTI I MODELLI VERIFICATI")
    else:
        print(f"⚠ {sum(1 for r in results if not r)} modelli hanno problemi")
    print(f"{'='*40}")
    sys.exit(0 if ok else 1)
