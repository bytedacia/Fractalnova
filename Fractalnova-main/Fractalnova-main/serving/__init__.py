"""FractalNova · model serving tier (vLLM, OpenAI-compatible).

Disaccoppia la generazione pesante (GPU) dall'API. L'API parla con questo tier
via HTTP OpenAI-compatible; se non configurato, degrada con grazia.
"""
__version__ = "0.1.0"
