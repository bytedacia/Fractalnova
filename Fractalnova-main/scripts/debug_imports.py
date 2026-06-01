"""Debug: prova a importare i moduli chiave e classifica i problemi.

OK  = importa senza errori
DEP = manca una dipendenza opzionale (non un bug del codice)
ERR = errore reale nel codice (da sistemare)
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODULES = [
    "fractalnova.config", "fractalnova.logging_config", "fractalnova.seo",
    "fractalnova.export", "fractalnova.publishing",
    "inference.calm", "inference.fractalnova", "inference.orchestrator", "inference.generate",
    "api.schemas", "api.db", "api.security", "api.services", "api.main",
    "serving.client", "serving.generation",
]

ok, dep, err = [], [], []
for m in MODULES:
    try:
        importlib.import_module(m)
        ok.append(m)
        print(f"OK    {m}")
    except ModuleNotFoundError as e:
        dep.append((m, e.name))
        print(f"DEP   {m:32s} manca: {e.name}")
    except Exception as e:
        err.append((m, f"{type(e).__name__}: {e}"))
        print(f"ERR   {m:32s} {type(e).__name__}: {e}")

print("\n===== RIEPILOGO =====")
print(f"OK : {len(ok)}   DEP(dipendenze mancanti): {len(dep)}   ERR(bug): {len(err)}")
if dep:
    print("Dipendenze mancanti:", sorted({d[1] for d in dep}))
if err:
    print("\nBUG DA SISTEMARE:")
    for m, e in err:
        print(f"  {m}: {e}")
