"""Test dell'export (txt/epub in puro Python, senza dipendenze pesanti)."""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fractalnova.export import export_epub, export_txt, iter_chapters  # noqa: E402

BOOK_CONTENT = {
    "title": "Test Libro",
    "plot": "Una trama breve.",
    "chapters": [{"title": "Capitolo 1", "content": "Primo paragrafo.\n\nSecondo paragrafo."}],
}
BOOK_SUBCHAPTERS = {
    "title": "Test Sub",
    "chapters": [{"title": "Cap 1", "subchapters": [{"title": "Sez A", "content": "Testo A."}]}],
}


def test_iter_chapters_handles_flat_content():
    chapters = list(iter_chapters(BOOK_CONTENT))
    assert chapters[0][0] == "Capitolo 1"
    assert "Primo paragrafo" in chapters[0][1]


def test_iter_chapters_handles_subchapters():
    chapters = list(iter_chapters(BOOK_SUBCHAPTERS))
    assert "Testo A." in chapters[0][1]
    assert "Sez A" in chapters[0][1]


def test_export_txt(tmp_path):
    path = export_txt(BOOK_CONTENT, out_dir=str(tmp_path))
    assert os.path.exists(path)
    assert "Capitolo 1" in open(path, encoding="utf-8").read()


def test_export_epub_is_valid_zip_with_mimetype_first(tmp_path):
    path = export_epub(BOOK_CONTENT, out_dir=str(tmp_path))
    assert os.path.exists(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert names[0] == "mimetype"  # requisito EPUB
        assert z.read("mimetype").decode() == "application/epub+zip"
        assert "OEBPS/content.opf" in names
        assert "OEBPS/nav.xhtml" in names
        assert any(n.startswith("OEBPS/chap_") for n in names)
