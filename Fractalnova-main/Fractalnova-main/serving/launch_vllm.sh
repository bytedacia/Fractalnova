#!/usr/bin/env bash
# Avvia il tier modello FractalNova-Pro con vLLM (endpoint OpenAI-compatible).
# Uso: ./serving/launch_vllm.sh [percorso_modello]
set -euo pipefail

MODEL_PATH="${1:-training/outputs/fractalnova-pro-merged}"
PORT="${VLLM_PORT:-8001}"

echo "[vLLM] avvio modello: ${MODEL_PATH} su :${PORT}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name fractalnova-pro \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
