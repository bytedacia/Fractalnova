import os
from typing import Dict

from .fractalnova import FractalNovaInference


class FractalNova:
    """Orchestratore end-to-end che usa FractalNovaInference come unico motore AI."""

    def __init__(self, profile: str = "pro") -> None:
        self.ai = FractalNovaInference(profile=profile)
        self.author_name = os.getenv("AUTHOR_NAME", "")
        self.author_email = os.getenv("AUTHOR_EMAIL", "")
        self._profile = profile

    def run(self, book_details: Dict) -> Dict:
        book = self.ai.generate_book(book_details)
        seo = self.ai.analyze_seo(book.get("full_text", ""))
        book["seo"] = seo

        from .generate import save_as_word, wattpad_export
        word_path = save_as_word(book, book.get("title", "libro"))
        wattpad_path = wattpad_export(book)
        book["word_path"] = word_path
        book["wattpad_path"] = wattpad_path

        cover = self.ai.generate_cover(
            book_details.get("title", ""),
            book_details.get("genre", ""),
            context=book.get("full_text", "")[:500],
        )
        book["cover_path"] = cover

        return {
            "book_structure": book,
            "word_path": word_path,
            "wattpad_path": wattpad_path,
            "cover_path": cover,
            "seo": seo,
        }
