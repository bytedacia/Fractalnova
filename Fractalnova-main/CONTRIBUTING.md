# Contribuire a FractalNova

Grazie per l'interesse. Questo progetto segue pratiche da team professionale.

## Setup di sviluppo
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # pacchetto + strumenti di sviluppo
pip install -e ".[train]"      # opzionale: dipendenze di training
```

## Prima di aprire una PR
```bash
ruff check .        # lint
black --check .     # formattazione
pytest              # test (quelli che richiedono torch si saltano da soli)
```

## Convenzioni
- **Stile**: `black` + `ruff`, riga max 120.
- **Commit**: messaggi chiari e in imperativo (es. "Aggiungi export EPUB").
- **Branch**: parti da `main`, un branch per feature/fix.
- **Sicurezza**: non committare segreti. Usa `.env` (vedi `.env.example`); `.env` è in `.gitignore`.
- **Test**: ogni modulo con logica pura va con un test in `tests/`.

## Aree di lavoro (workstream)
Generation · Humanization · SEO · Cover · Export · Publishing · Frontend ·
Security · DevOps · QA/Docs · Model (Core/Pro). Vedi `README.md` per la mappa.

## Segnalazioni
Usa i template in `.github/ISSUE_TEMPLATE/`.
