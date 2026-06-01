# FractalNova API

Servizio **FastAPI** di produzione: auth, job di generazione asincroni, persistenza,
rate limiting, export multi-formato. Degrada con grazia senza GPU (utile in dev/CI).

## Avvio locale
```bash
pip install -r api/requirements-api.txt
uvicorn api.main:app --reload --port 8000
# Docs interattive: http://localhost:8000/docs
```

## Flusso (curl)
```bash
# 1) registrazione
curl -X POST localhost:8000/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"password123"}'

# 2) login -> token
TOKEN=$(curl -s -X POST localhost:8000/auth/token \
  -d 'username=me@example.com&password=password123' | jq -r .access_token)

# 3) avvia generazione (async) -> job
JOB=$(curl -s -X POST localhost:8000/v1/books -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Il Sale degli Dei","genre":"fantasy","target_pages":120}' | jq -r .id)

# 4) stato job -> book_id
curl -s localhost:8000/v1/jobs/$JOB -H "Authorization: Bearer $TOKEN"

# 5) libro + formati di export
curl -s localhost:8000/v1/books/1 -H "Authorization: Bearer $TOKEN"
```

## Endpoint
| Metodo | Path | Auth | Descrizione |
|---|---|---|---|
| GET | `/health` | no | Stato servizio |
| POST | `/auth/register` | no | Crea utente |
| POST | `/auth/token` | no | Login OAuth2 → JWT |
| GET | `/v1/me` | sì | Profilo utente |
| POST | `/v1/books` | sì | Avvia job di generazione (202) |
| GET | `/v1/jobs/{id}` | sì | Stato job |
| GET | `/v1/books/{id}` | sì | Libro + export |

## Docker / Kubernetes
```bash
docker build -f api/Dockerfile -t fractalnova-api .
docker run -p 8000:8000 -e API_JWT_SECRET=change-me fractalnova-api

kubectl create secret generic fractalnova-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  --from-literal=database-url='postgresql+psycopg2://user:pass@db:5432/fractalnova'
kubectl apply -f api/k8s/
```

## Architettura (2 tier)
```
            ┌──────────────────────┐        ┌───────────────────────────┐
  Client →  │  API tier (FastAPI)  │  gRPC/ │  Model tier (vLLM, GPU)   │
            │  auth, job, billing  │  HTTP →│  FractalNova-Pro / DeepSeek│
            │  CPU, autoscalato    │        │  batching, KV-cache        │
            └──────────┬───────────┘        └───────────────────────────┘
                       │ Postgres (utenti/libri/job)  │ Redis (rate limit/cache)
```
Il tier API è **stateless** e scala in orizzontale (HPA 3→50 pod). La generazione
pesante vive nel tier modello (GPU), disaccoppiato via job asincroni.

## Sicurezza
JWT (HS256), password bcrypt, rate limiting per utente, CORS configurabile,
isolamento per `user_id` su job e libri. Segreti via K8s Secret / env, mai nel codice.
