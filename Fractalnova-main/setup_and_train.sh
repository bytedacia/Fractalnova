#!/usr/bin/env bash
#
# FractalNova-Pro · Setup & Training · One-command launcher
# ==========================================================
# Usage:  chmod +x setup_and_train.sh && ./setup_and_train.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── 1. Check environment ──────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "ERR: python3 required"; exit 1; }
echo "✓ Python $(python3 --version | cut -d' ' -f2)"

if python3 -c "import torch; print('✓ CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" 2>/dev/null; then
    :
else
    echo "⚠ torch non installato — verrà installato ora"
fi

# ── 2. Create venv & install deps ────────────────────────────────
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── 3. Complete model downloads (if missing) ──────────────────────
echo "── Verifica/Download modelli ──"
python3 -c "
import os, sys
from pathlib import Path

models = {
    'models/Qwen3-4B':    ('Qwen/Qwen3-4B', os.getenv('HF_TOKEN')),
    'models/gemma-4-E2B': ('google/gemma-4-E2B', os.getenv('HF_TOKEN')),
    'models/FLUX.1-dev':  ('black-forest-labs/FLUX.1-dev', os.getenv('HF_TOKEN')),
}

for local, (repo, token) in models.items():
    local = Path(local)
    # check if at least one .safetensors exists
    safetensors = list(local.rglob('*.safetensors'))
    if safetensors and any(s.stat().st_size > 1000000 for s in safetensors):
        print(f'  ✓ {local.name} OK ({len(safetensors)} safetensors files)')
        continue
    print(f'  ↓ Download {repo} → {local} …')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(local), token=token)
    print(f'  ✓ {local.name} completo')
"

# ── 4. Prepare dataset ────────────────────────────────────────────
echo "── Preparazione dataset ──"
DATA_DIR="training/data"
mkdir -p "$DATA_DIR"

# If no train.jsonl exists, prepare from sample data or generate
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    # Try sample files first
    if [ -f "$DATA_DIR/sample_books_it.jsonl" ] && [ -f "$DATA_DIR/sample_books_multi.jsonl" ]; then
        echo "  Usando dati sample (pochi esempi — genero dataset completo)"
        python3 training/prepare_dataset.py \
            --inputs "$DATA_DIR/sample_books_it.jsonl" "$DATA_DIR/sample_books_multi.jsonl" \
            --out-dir "$DATA_DIR" --val-ratio 0.2
    else
        echo "  Generazione dataset sintetico (NO API richiesta)..."
        python3 training/dataset_generator_noapi.py \
            --num-examples 50000 \
            --out-dir "$DATA_DIR/generated"
        python3 training/prepare_dataset.py \
            --inputs "$DATA_DIR/generated" \
            --out-dir "$DATA_DIR" --val-ratio 0.1
    fi
else
    echo "  ✓ $DATA_DIR/train.jsonl gia presente"
fi

# ── 5. Fine-tune (unified multi-model) ────────────────────────────
echo "── Training FractalNova-Pro ──"
python3 training/train_fractalnova_pro.py \
    --train-file "$DATA_DIR/train.jsonl" \
    --val-file "$DATA_DIR/val.jsonl" \
    --output-dir "training/outputs/fractalnova-pro" \
    --epochs 3

# ── 6. Verify ─────────────────────────────────────────────────────
echo "── Verifica modello fine-tunato ──"
python3 -c "
from inference.fractalnova import FractalNovaInference
ai = FractalNovaInference('pro')
# Force check for fine-tuned adapters
if ai._check_manifest():
    print('✓ FractalNova-Pro (fine-tunato) pronto')
else:
    print('⚠ Fine-tuning completato ma manifest non trovato')
    print('  Il modello base e comunque caricabile con FractalNovaInference(\"pro\")')

# Quick inference test
result = ai.generate('Scrivi due righe di un romanzo gotico.')
print(f'✓ Generazione OK ({len(result.get(\"text\",\"\"))} caratteri)')
print(f'  Anteprima: {result.get(\"text\", \"\")[:120]}...')
"

echo ""
echo "=== FractalNova-Pro training completato ==="
echo "Modello fine-tunato: training/outputs/fractalnova-pro/"
echo "Per usarlo:  python3 -c \"from inference.fractalnova import FractalNovaInference; ai = FractalNovaInference('pro'); print(ai.generate('Scrivi un incipit'))\""
