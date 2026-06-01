"""Test del parsing/normalizzazione SEO (puro, senza dipendenze pesanti)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fractalnova.seo import normalize_seo, parse_seo  # noqa: E402


def test_parse_clean_json():
    out = parse_seo('{"keywords": ["a", "b"], "tags": ["x"], "description": "d", "categories": ["Fiction"]}')
    assert out["keywords"] == ["a", "b"]
    assert out["categories"] == ["Fiction"]


def test_parse_dirty_json_with_surrounding_text():
    raw = 'Ecco il risultato:\n{"keywords": ["mare"], "description": "Una storia"}\nGrazie!'
    out = parse_seo(raw)
    assert out["keywords"] == ["mare"]
    assert out["description"] == "Una storia"


def test_description_is_clamped_to_160():
    out = normalize_seo({"description": "x" * 300})
    assert len(out["description"]) <= 160
    assert out["description"].endswith("...")


def test_keywords_from_csv_string():
    out = normalize_seo({"keywords": "uno, due, tre"})
    assert out["keywords"] == ["uno", "due", "tre"]


def test_empty_input_gives_stable_schema():
    out = parse_seo("nessun json qui")
    assert set(out.keys()) == {"keywords", "tags", "description", "categories"}
