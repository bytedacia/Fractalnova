#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────
# FractalNova CALM — one-command bridge training
# Usage: bash train_calm.sh [--anchor MODEL] [--augmenting MODEL] [options]
# ─────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
ANCHOR="Qwen/Qwen3-4B"
AUGMENTING="HuggingFaceTB/SmolLM2-1.7B-Instruct"
BRIDGE_LAYERS="8 16 24"
LR="2e-4"
BATCH_SIZE="4"
GRAD_ACCUM="8"
MAX_STEPS="5000"
OUTPUT_DIR="calm_output"
TRAIN_FILE=""
EVAL_FILE=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anchor) ANCHOR="$2"; shift 2 ;;
    --augmenting) AUGMENTING="$2"; shift 2 ;;
    --bridge-layers) BRIDGE_LAYERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --train-file) TRAIN_FILE="$2"; shift 2 ;;
    --eval-file) EVAL_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$TRAIN_FILE" ]]; then
  echo "Error: --train-file is required (JSONL with 'text' field)"
  exit 1
fi

# Check venv
if [[ -d "venv" ]]; then
  source venv/bin/activate
elif [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

# Check dependencies
python -c "import torch, transformers, accelerate, datasets" 2>/dev/null || {
  echo "Installing dependencies..."
  pip install torch transformers accelerate datasets tensorboard --upgrade
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         FractalNova CALM Bridge Training            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Anchor:      $ANCHOR"
echo "  Augmenting:  $AUGMENTING"
echo "  Bridge:      $BRIDGE_LAYERS"
echo "  Steps:       $MAX_STEPS"
echo "  Batch:       $BATCH_SIZE x $GRAD_ACCUM"
echo "  Output:      $OUTPUT_DIR"
echo "  Train:       $TRAIN_FILE"
echo ""

python train_calm.py \
  --anchor "$ANCHOR" \
  --augmenting "$AUGMENTING" \
  --bridge-layers $BRIDGE_LAYERS \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  --train-file "$TRAIN_FILE" \
  ${EVAL_FILE:+--eval-file "$EVAL_FILE"}

echo ""
echo "✓ CALM training complete — bridge weights saved to $OUTPUT_DIR"
