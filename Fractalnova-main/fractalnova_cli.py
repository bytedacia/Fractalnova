"""
FractalNova · CLI end-to-end.

Usa FractalNovaInference come unico motore AI (Pro: Qwen3-4B, Core: 124M).
Pipeline completa: scrittura -> umanizzazione -> SEO -> copertina FLUX.1-dev
-> export Word+EPUB+PDF -> pacchetto KDP -> outreach editori.

Esempi:
    python fractalnova_cli.py --title "Il Sale degli Dei" --genre fantasy --pages 120
    python fractalnova_cli.py --from-json book.json
"""
import argparse
import json

from dotenv import load_dotenv


def build_details(args) -> dict:
    return {
        "title": args.title or "Senza Titolo",
        "genre": args.genre,
        "plot": args.plot,
        "chapter_structure": args.structure,
        "target_pages": args.pages,
    }


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="FractalNova · genera un libro end-to-end")
    ap.add_argument("--from-json", help="file JSON con i dettagli del libro")
    ap.add_argument("--title")
    ap.add_argument("--genre", default="fiction")
    ap.add_argument("--plot", default="")
    ap.add_argument("--structure", default="misto", choices=["breve", "lungo", "misto"])
    ap.add_argument("--pages", type=int, default=100)
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            book_details = json.load(f)
    else:
        book_details = build_details(args)

    # default richiesti dall'orchestratore
    book_details.setdefault("chapter_structure", "misto")
    book_details.setdefault("target_pages", 100)
    book_details.setdefault("title", "Senza Titolo")
    book_details.setdefault("genre", "fiction")

    # import ritardato: carica torch/transformers solo quando serve davvero
    from inference.orchestrator import FractalNova

    print("[FractalNova] avvio pipeline end-to-end...")
    engine = FractalNova()
    result = engine.run(book_details)

    bs = result.get("book_structure", {})

    # Export multi-formato (EPUB/PDF/Word/Wattpad) + pacchetto di sottomissione KDP
    from fractalnova.export import export_all
    from fractalnova.publishing import build_submission_package

    exports = export_all(bs, cover_path=result.get("cover_path"))
    package = build_submission_package(bs, assets={**exports, "cover": result.get("cover_path")})

    print("\n=== FractalNova · risultato ===")
    print("Titolo    :", bs.get("title"))
    print("Capitoli  :", len(bs.get("chapters", [])))
    print("Export    :", json.dumps({k: v for k, v in exports.items() if v}, ensure_ascii=False))
    print("Copertina :", result.get("cover_path"))
    print("Pacchetto :", package)
    print("SEO       :", json.dumps(result.get("seo", {}), ensure_ascii=False))
    print("Outreach  :", json.dumps(result.get("outreach", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
