"""Validazione dei dataset di esempio (JSONL valido + copertura multilingua)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "training", "data")


def test_sample_files_are_valid_jsonl():
    files = glob.glob(os.path.join(DATA, "sample_*.jsonl"))
    assert files, "Nessun file di esempio trovato in training/data"
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    json.loads(line)  # solleva se JSON non valido


def test_multilingual_sample_covers_several_languages():
    fp = os.path.join(DATA, "sample_books_multi.jsonl")
    langs = set()
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                langs.add(json.loads(line).get("lang"))
    assert {"en", "es", "fr", "de"}.issubset(langs)


def test_multilingual_sample_covers_several_tasks():
    fp = os.path.join(DATA, "sample_books_multi.jsonl")
    tasks = set()
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.add(json.loads(line).get("task"))
    # scrittura/riscrittura/seo/titolo/sinossi/traduzione: deve esserci varieta'
    assert len(tasks) >= 4
