import os
from dotenv import load_dotenv
from inference.fractalnova import FractalNovaInference
import gradio as gr

load_dotenv()

try:
    from security.input_sanitizer import sanitize_prompt, sanitize_title
    from security.prompt_guard import check_prompt_injection as guard_injection
    from security.output_sanitizer import sanitize_model_output
    from security.model_rate_limiter import check_model_rate_limit, record_model_call
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

ai = FractalNovaInference(profile=os.getenv("FRACTALNOVA_PROFILE", "pro"))

HEAD = ""
if SECURITY_AVAILABLE:
    from security.security_headers import gradio_head_html
    HEAD = gradio_head_html()

APP_TITLE = "FractalNova - Generazione libri con IA"


def _secure(prompt: str) -> str:
    if SECURITY_AVAILABLE:
        prompt = sanitize_prompt(prompt or "")
        safe, reason, _ = guard_injection(prompt)
        if not safe:
            return f"[Sicurezza] Input non consentito: {reason}"
        allowed, remaining, retry = check_model_rate_limit("default")
        if not allowed:
            return f"[Sicurezza] Troppe richieste. Riprova tra {retry} secondi."
        record_model_call("default")
    return prompt


def generate_text(prompt, style, content_type, temperature, max_new_tokens):
    prompt = _secure(prompt)
    if prompt.startswith("[Sicurezza]"):
        return prompt
    out = ai.generate(prompt, temperature=temperature, max_tokens=max_new_tokens)
    return sanitize_model_output(out) if SECURITY_AVAILABLE else out


def analyze_style(text):
    text = _secure(text)
    if text.startswith("[Sicurezza]"):
        return text
    out = ai.analyze_style(text)
    return sanitize_model_output(str(out)) if SECURITY_AVAILABLE else str(out)


def generate_book(title, genre, style, content_type, temperature, max_new_tokens):
    if SECURITY_AVAILABLE:
        title = sanitize_title(title or "")
    out = ai.generate(
        f"Scrivi un libro intitolato '{title}' nel genere {genre}. Stile: {style}.",
        temperature=temperature, max_tokens=max_new_tokens,
    )
    return sanitize_model_output(out) if SECURITY_AVAILABLE else out


def generate_cover_ui(title, genre, cover_style, theme):
    if SECURITY_AVAILABLE:
        title = sanitize_title(title or "")
    path = ai.generate_cover(title, genre, style=cover_style, theme=theme)
    return path if path and os.path.exists(path) else None


def analyze_cover_ui(image_path):
    if not image_path or not os.path.exists(image_path):
        return "Carica un'immagine di copertina prima di analizzare."
    desc = ai.describe_image(image_path)
    seo = ai.analyze_image_seo(image_path)
    genre = ai.classify_genre_from_cover(image_path)
    return (
        f"### Descrizione\n{desc}\n\n"
        f"### Genere rilevato\n{genre}\n\n"
        f"### SEO\n{seo}"
    )


with gr.Blocks(title=APP_TITLE, head=HEAD) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("*FractalNova Pro · Qwen3-4B · Gemma-4-E2B · FLUX.1-dev · un'unica IA*")

    with gr.Tab("Generazione Testo"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Inserisci il tuo prompt qui...", max_lines=5)
                style = gr.Dropdown(["naturale", "formale", "creativo", "tecnico", "narrativo"], label="Stile", value="naturale")
                content_type = gr.Dropdown(["qualsiasi", "articolo", "storia", "poesia", "saggio", "romanzo"], label="Tipo di Contenuto", value="qualsiasi")
                temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperatura")
                max_new_tokens = gr.Slider(minimum=32, maximum=131072, step=32, value=2048, label="Max nuovi token")
                generate_btn = gr.Button("Genera")
            with gr.Column():
                output = gr.Textbox(label="Risultato", lines=10)
        generate_btn.click(generate_text, inputs=[prompt, style, content_type, temperature, max_new_tokens], outputs=output)

    with gr.Tab("Analisi Stile"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Testo da Analizzare", lines=5)
                analyze_btn = gr.Button("Analizza")
            with gr.Column():
                analysis_output = gr.Textbox(label="Analisi", lines=10)
        analyze_btn.click(analyze_style, inputs=text_input, outputs=analysis_output)

    with gr.Tab("Generazione Libro"):
        with gr.Row():
            with gr.Column():
                title = gr.Textbox(label="Titolo del libro", placeholder="Es: Il viaggio segreto")
                genre = gr.Dropdown(["fiction", "non-fiction", "fantasy", "sci-fi", "thriller", "romance", "giallo", "storico"], label="Genere", value="fiction")
                book_style = gr.Dropdown(["naturale", "formale", "creativo", "tecnico", "narrativo"], label="Stile", value="naturale")
                book_content_type = gr.Dropdown(["qualsiasi", "romanzo", "racconto", "saggio", "novella"], label="Tipo di Contenuto", value="qualsiasi")
                book_temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperatura")
                book_max_new_tokens = gr.Slider(minimum=128, maximum=131072, step=64, value=4096, label="Max nuovi token")
                book_generate_btn = gr.Button("Genera Libro")
            with gr.Column():
                book_output = gr.Textbox(label="Libro Generato", lines=15)
        book_generate_btn.click(
            generate_book,
            inputs=[title, genre, book_style, book_content_type, book_temperature, book_max_new_tokens],
            outputs=book_output,
        )

    with gr.Tab("Copertina (FLUX.1-dev)"):
        with gr.Row():
            with gr.Column():
                cover_title = gr.Textbox(label="Titolo del libro", placeholder="Es: Il Giardino delle Esperidi")
                cover_genre = gr.Dropdown(
                    ["fiction", "non-fiction", "fantasy", "sci-fi", "thriller", "romance", "giallo", "storico", "horror", "avventura"],
                    label="Genere", value="fantasy"
                )
                cover_style = gr.Dropdown(
                    ["realistico", "minimalista", "onirico", "vintage", "cinematografico", "acquerello", "grafico vettoriale"],
                    label="Stile copertina", value="cinematografico"
                )
                cover_theme = gr.Textbox(label="Tema / elementi visivi", placeholder="Es: astronave, giungla, colori freddi, 8k")
                cover_generate_btn = gr.Button("Genera Copertina")
            with gr.Column():
                cover_output = gr.Image(label="Copertina Generata", type="filepath")
        cover_generate_btn.click(generate_cover_ui, inputs=[cover_title, cover_genre, cover_style, cover_theme], outputs=cover_output)

    with gr.Tab("Analisi Copertina (Gemma-4-E2B)"):
        with gr.Row():
            with gr.Column():
                analyze_image = gr.Image(label="Carica copertina", type="filepath")
                analyze_cover_btn = gr.Button("Analizza Copertina")
            with gr.Column():
                analysis_result = gr.Markdown(label="Analisi Visiva")
        analyze_cover_btn.click(analyze_cover_ui, inputs=analyze_image, outputs=analysis_result)

SERVER_PORT = int(os.getenv("FRACTALNOVA_PORT", "7860"))
SERVER_NAME = os.getenv("FRACTALNOVA_HOST", "0.0.0.0")
demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
