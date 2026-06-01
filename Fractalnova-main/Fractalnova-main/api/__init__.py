"""FractalNova · API di produzione (FastAPI).

Servizio enterprise per la generazione di libri: auth JWT, job asincroni,
persistenza, rate limiting, export multi-formato. Degrada con grazia se i
modelli pesanti non sono disponibili (utile in CI e in dev senza GPU).
"""
__version__ = "0.1.0"
