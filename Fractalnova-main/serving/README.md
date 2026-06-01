# FractalNova · Model Serving (vLLM)

Tier modello disaccoppiato: serve **FractalNova-Pro** via endpoint OpenAI-compatible.
L'API (`api/`) lo usa se `MODEL_SERVER_URL` è impostato; altrimenti degrada.

## Locale (GPU)
```bash
pip install vllm
./serving/launch_vllm.sh training/outputs/fractalnova-pro-merged
# Endpoint: http://localhost:8001/v1
```

## Collegare l'API al tier modello
```bash
export MODEL_SERVER_URL=http://localhost:8001/v1
export MODEL_SERVER_NAME=fractalnova-pro
uvicorn api.main:app --port 8000
# Ora /v1/books genera col modello reale (non più segnaposto).
```

## Kubernetes (GPU)
```bash
kubectl apply -f serving/k8s/vllm-deployment.yaml
# l'API punta al service interno:
#   MODEL_SERVER_URL=http://fractalnova-vllm:8001/v1
```

## Perché vLLM
- **Throughput**: continuous batching + PagedAttention → molte più req/s per GPU.
- **OpenAI-compatible**: zero lock-in, client standard.
- **Scalabile**: repliche dietro un service; KV-cache efficiente.

## Catena di fallback (in `api/services.py`)
1. **Model tier (vLLM)** se `MODEL_SERVER_URL` configurato → generazione reale.
2. **Orchestratore locale** se i pesi sono su questa macchina.
3. **Segnaposto** altrimenti (l'API non crasha mai).
