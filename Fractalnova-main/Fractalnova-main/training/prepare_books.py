"""
FractalNova · ingestione di una cartella di LIBRI -> dataset di training (stile).

Crea coppie di CONTINUAZIONE (contesto precedente -> prosegui) per insegnare al
modello lo STILE e la coerenza narrativa dei tuoi libri.

Formati:
  - .parquet (dataset tabellari: auto-rileva la colonna testo; usa --inspect per lo schema)
  - .txt/.md nativi
  - .epub/.pdf/.docx con dipendenze opzionali: pip install ebooklib beautifulsoup4 pypdf python-docx
  - .parquet richiede: pip install pyarrow

Uso:
    python training/prepare_books.py --inspect                       # mostra schema dei parquet
    python training/prepare_books.py --books-dir training/data/books_raw --out training/data/books.jsonl
    python training/prepare_books.py --text-column text --group-column title
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

SYSTEM = (
    "Sei FractalNova, autore ed editor professionista. Scrivi prosa narrativa coerente "
    "e naturale nella lingua del testo, mantenendo stile, voce, ritmo e continuità."
)

TEXT_COL_CANDIDATES = ["text", "content", "body", "page_content", "markdown", "chapter",
                       "document", "book", "raw", "story", "testo", "contenuto"]
GROUP_COL_CANDIDATES = ["title", "book_id", "book", "source", "doc_id", "id", "filename", "titolo"]


# --------------------------------------------------------------------------- #
# Lettori file testuali
# --------------------------------------------------------------------------- #
def read_txt(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_epub(path):
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        from ebooklib import epub
    except Exception:
        return None
    book = epub.read_epub(path)
    return "\n".join(BeautifulSoup(it.get_content(), "html.parser").get_text(" ")
                     for it in book.get_items_of_type(ebooklib.ITEM_DOCUMENT))


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)


def read_docx(path):
    try:
        import docx
    except Exception:
        return None
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


FILE_READERS = {".txt": read_txt, ".md": read_txt, ".epub": read_epub, ".pdf": read_pdf, ".docx": read_docx}


# --------------------------------------------------------------------------- #
# Parquet
# --------------------------------------------------------------------------- #
def _read_parquet(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.column_names, table.to_pylist()


def _detect_text_col(cols, rows):
    low = {c.lower(): c for c in cols}
    for cand in TEXT_COL_CANDIDATES:
        if cand in low:
            return low[cand]
    best, best_len = None, -1.0
    for c in cols:
        vals = [r.get(c) for r in rows[:100] if isinstance(r.get(c), str)]
        if vals:
            avg = sum(len(v) for v in vals) / len(vals)
            if avg > best_len:
                best, best_len = c, avg
    return best


def _detect_group_col(cols):
    low = {c.lower(): c for c in cols}
    for cand in GROUP_COL_CANDIDATES:
        if cand in low:
            return low[cand]
    return None


def parquet_docs(paths, text_col=None, group_col=None, inspect=False):
    docs = []
    for path in paths:
        cols, rows = _read_parquet(path)
        if inspect:
            print(f"[parquet] {os.path.basename(path)} | righe: {len(rows)} | colonne: {cols}")
            if rows:
                sample = {k: (str(v)[:100] + ("…" if v and len(str(v)) > 100 else "")) for k, v in rows[0].items()}
                print(f"          riga 0: {sample}")
                print(f"          colonna testo rilevata: {_detect_text_col(cols, rows)} | gruppo: {_detect_group_col(cols)}")
            continue
        tc = text_col or _detect_text_col(cols, rows)
        if not tc:
            raise SystemExit(f"Nessuna colonna testo in {cols}. Usa --text-column.")
        gc = group_col or _detect_group_col(cols)
        if gc and gc in cols:
            groups = {}
            for r in rows:
                groups.setdefault(r.get(gc), []).append(str(r.get(tc) or ""))
            for parts in groups.values():
                docs.append("\n\n".join(p for p in parts if p))
        else:
            docs.append("\n\n".join(str(r.get(tc) or "") for r in rows))
        print(f"[parquet] {os.path.basename(path)}: colonna '{tc}'"
              + (f", gruppo '{gc}'" if gc else "") + f" -> {len(docs)} doc finora")
    return docs


# --------------------------------------------------------------------------- #
def clean(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text, words=600, overlap=80):
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + words]))
        i += max(1, words - overlap)
    return out


def gather_docs(books_dir, text_col, group_col, inspect):
    docs = []
    parquets = glob.glob(os.path.join(books_dir, "**", "*.parquet"), recursive=True)
    if parquets:
        docs += parquet_docs(parquets, text_col, group_col, inspect)
    if inspect:
        return []
    for ext, reader in FILE_READERS.items():
        for fp in glob.glob(os.path.join(books_dir, "**", f"*{ext}"), recursive=True):
            t = reader(fp)
            if t:
                docs.append(t)
            else:
                print(f"[skip] {os.path.basename(fp)} (manca la dipendenza per {ext})")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-dir", default="training/data/books_raw")
    ap.add_argument("--out", default="training/data/books.jsonl")
    ap.add_argument("--words-per-chunk", type=int, default=600)
    ap.add_argument("--text-column", default=None, help="[parquet] nome colonna testo")
    ap.add_argument("--group-column", default=None, help="[parquet] colonna per raggruppare i libri")
    ap.add_argument("--inspect", action="store_true", help="mostra solo lo schema dei .parquet")
    args = ap.parse_args()

    if not os.path.isdir(args.books_dir):
        raise SystemExit(f"Cartella inesistente: {args.books_dir}. Mettici i libri (.parquet/.txt/...).")

    docs = gather_docs(args.books_dir, args.text_column, args.group_column, args.inspect)
    if args.inspect:
        return
    if not docs:
        raise SystemExit(f"Nessun documento estratto da {args.books_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            cks = chunk(clean(d), args.words_per_chunk)
            for j in range(1, len(cks)):
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Prosegui il romanzo in modo coerente con quanto precede:\n\n{cks[j-1][-1200:]}"},
                    {"role": "assistant", "content": cks[j]},
                ]
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                rows += 1
    print(f"\n[books] {len(docs)} documenti -> {rows} esempi di training -> {args.out}")


if __name__ == "__main__":
    main()
