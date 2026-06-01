"""Pacchetto di inferenza di FractalNova (import lazy per robustezza)."""


def __getattr__(name):
    # Re-export lazy dei simboli di fractalnova.py: non rompe l'import del
    # pacchetto se manca un nome o una dipendenza opzionale.
    from . import fractalnova
    return getattr(fractalnova, name)
