"""
FractalNova · scarica CLASSICI FAMOSI MULTILINGUA da Project Gutenberg (pubblico dominio).

Testi corretti a mano (no OCR sporco). Rimuove header/footer Gutenberg.
Salva .txt in training/data/books_raw/ -> pronti per prepare_books.py.
Gli ID che falliscono vengono saltati (nessun crash).
"""
import os
import re
import time

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import httpx

# Classici FAMOSI per lingua — Project Gutenberg ebook IDs (pubblico dominio)
BOOKS = {
    "it": [3747, 47786, 49626, 44797, 46957, 38637, 48361, 48779, 57787, 7459,
           7267, 27359, 64766, 25178, 19517, 34518, 30030, 30663, 65391, 43226,
           10502, 25182, 30771, 22504, 57082, 17876, 25177, 48445, 17837],
    "en": [1342, 84, 1661, 2701, 11, 345, 98, 1400, 1260, 174,
           2600, 1399, 2554, 16, 768, 158, 1232, 219, 1184, 2542],
    "fr": [5781, 11049, 42256, 13848, 48212, 34469, 20577, 36826, 35568, 4650],
    "de": [5200, 5072, 5322, 21149, 31538, 2190, 2189, 8803, 63465, 50965, 8085],
    "es": [2000, 17073, 61202, 23600, 15725, 14944, 17341, 16413, 58298, 26983, 60464, 60284],
}

OUT = "training/data/books_raw"


def fetch(book_id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            r = httpx.get(url, timeout=60, follow_redirects=True,
                          headers={"User-Agent": "FractalNova/1.0 (research; public-domain texts)"})
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except Exception:
            continue
    return None


def strip_gutenberg(text):
    m = re.search(r"\*\*\*\s*START OF TH[EI][S ].*?PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF TH[EI][S ].*?PROJECT GUTENBERG", text, re.IGNORECASE)
    if m:
        text = text[:m.start()]
    return text.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = 0
    total_chars = 0
    by_lang = {}
    for lang, ids in BOOKS.items():
        for book_id in ids:
            text = fetch(book_id)
            if not text:
                print(f"  [{lang}] skip {book_id}")
                continue
            text = strip_gutenberg(text)
            if len(text) < 2000:
                print(f"  [{lang}] skip {book_id} (corto)")
                continue
            with open(os.path.join(OUT, f"gutenberg_{lang}_{book_id}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            ok += 1
            total_chars += len(text)
            by_lang[lang] = by_lang.get(lang, 0) + 1
            print(f"  [{lang}] ok {book_id}: {len(text):,} car.")
            time.sleep(0.8)
    print(f"\n[gutenberg] {ok} libri, {total_chars:,} caratteri -> {OUT}")
    print(f"[gutenberg] per lingua: {by_lang}")


if __name__ == "__main__":
    main()
