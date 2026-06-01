# FractalNova API · immagine di produzione (tier applicativo, CPU).
# La generazione pesante viene delegata al tier modello (vLLM); se irraggiungibile,
# l'API degrada al generatore segnaposto. Vedi api/README.md (architettura).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements-api.txt ./api/requirements-api.txt
RUN pip install --upgrade pip && pip install -r api/requirements-api.txt

# Il servizio usa il pacchetto fractalnova/ (export/SEO)
COPY fractalnova ./fractalnova
COPY api ./api

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
