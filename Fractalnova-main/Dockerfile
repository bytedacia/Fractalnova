# FractalNova · immagine applicativa (parte "prodotto").
# NB: questa immagine e' CPU. Per l'inferenza su GPU usare una base CUDA
#     (es. nvidia/cuda:12.8-runtime) e installare torch con l'index cu128.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# Dipendenze di sistema minime (alcuni wheel le richiedono)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Utente non-root (buona pratica di sicurezza)
RUN useradd --create-home appuser && chown -R appuser /code
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
