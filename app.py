import os
from dotenv import load_dotenv
import google.generativeai as genai
import torch
from transformers import AutoTokenizer
from inference.model import Transformer, ModelArgs
from inference.generate import generate
import json
import gradio as gr

# Sicurezza pipeline: header, sanitizzazione, prompt guard, rate limit, output
try:
    from security.security_headers import gradio_head_html
    from security.input_sanitizer import sanitize_prompt, sanitize_title
    from security.prompt_guard import check_prompt_injection as guard_injection
    from security.output_sanitizer import sanitize_model_output
    from security.model_rate_limiter import check_model_rate_limit, record_model_call
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione API Google
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Configurazione del modello Gemini
gemini_model = genai.GenerativeModel('gemini-pro')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazione del modello DeepSeek (lazy loading + fallback)
_deepseek_model = None
_tokenizer = None

def _load_deepseek_model(model_path: str, config_path: str):
    global _deepseek_model, _tokenizer
    if _deepseek_model is not None and _tokenizer is not None:
        return _deepseek_model, _tokenizer
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            args = ModelArgs(**json.load(f))
        torch.set_default_dtype(torch.bfloat16)
        if device.type == "cuda":
            torch.set_default_device("cuda")
        _deepseek_model = Transformer(args).to(device)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        _deepseek_model, _tokenizer = None, None
    return _deepseek_model, _tokenizer

# Branding e percorsi FractalNova
PROJECT_NAME = "FractalNova"
APP_TITLE = f"{PROJECT_NAME} - Generazione libri con IA"
DEFAULT_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", os.getenv("DEESEEK_MODEL_PATH", "models/deepseek-coder-7b"))
DEFAULT_CONFIG_PATH = os.getenv("DEEPSEEK_CONFIG_PATH", os.getenv("DEESEEK_CONFIG_PATH", "inference/configs/config_7B.json"))

def generate_text(prompt, style, content_type, temperature, max_new_tokens):
    # Sicurezza: sanitizzazione e prompt injection
    if SECURITY_AVAILABLE:
        prompt = sanitize_prompt(prompt or "")
        safe, reason, _ = guard_injection(prompt)
        if not safe:
            return f"[Sicurezza] Input non consentito: {reason}"
        allowed, remaining, retry = check_model_rate_limit("default")
        if not allowed:
            return f"[Sicurezza] Troppe richieste. Riprova tra {retry} secondi."
        record_model_call("default")
    # Gemini
    gemini_response = gemini_model.generate_content(prompt)

    # DeepSeek (best-effort)
    ds_text = None
    model, tok = _load_deepseek_model(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH)
    if model is not None and tok is not None:
        try:
            input_ids = tok.encode(prompt)
            out_ids = generate(
                model,
                [input_ids],
                int(max_new_tokens),
                tok.eos_token_id if tok.eos_token_id is not None else -1,
                float(temperature),
            )[0]
            ds_text = tok.decode(out_ids)
        except Exception:
            ds_text = None

    out = gemini_response.text or ""
    if ds_text:
        out = f"{out}\n\n{ds_text}"
    if SECURITY_AVAILABLE:
        out = sanitize_model_output(out)
    return out

def analyze_style(text):
    if SECURITY_AVAILABLE:
        text = sanitize_prompt(text or "")
        safe, reason, _ = guard_injection(text)
        if not safe:
            return f"[Sicurezza] Input non consentito: {reason}"
    prompt = f"Analizza lo stile del seguente testo e fornisci suggerimenti per migliorarlo:\n\n{text}"
    response = gemini_model.generate_content(prompt)
    out = response.text or ""
    if SECURITY_AVAILABLE:
        out = sanitize_model_output(out)
    return out

def generate_book(title, genre, style, content_type, temperature, max_new_tokens):
    if SECURITY_AVAILABLE:
        title = sanitize_title(title or "")
    prompt = (
        f"Scrivi un libro intitolato '{title}' nel genere {genre}. "
        f"Stile: {style}. Tipo di contenuto: {content_type}"
    )
    return generate_text(prompt, style, content_type, temperature, max_new_tokens)

# Creazione dell'interfaccia Gradio (header sicurezza: CSP, X-Frame-Options, etc.)
_head = gradio_head_html() if SECURITY_AVAILABLE else ""
with gr.Blocks(title=APP_TITLE, head=_head) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("*Scrivi testi, analizza stile e genera libri con intelligenza artificiale.*")
    
    with gr.Tab("Generazione Testo"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Inserisci il tuo prompt qui...", max_lines=5)
                style = gr.Dropdown(["naturale", "formale", "creativo", "tecnico", "narrativo"], label="Stile", value="naturale")
                content_type = gr.Dropdown(["qualsiasi", "articolo", "storia", "poesia", "saggio", "romanzo"], label="Tipo di Contenuto", value="qualsiasi")
                temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperatura")
                max_new_tokens = gr.Slider(minimum=32, maximum=2048, step=32, value=256, label="Max nuovi token")
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
                book_max_new_tokens = gr.Slider(minimum=128, maximum=8192, step=64, value=1024, label="Max nuovi token")
                book_generate_btn = gr.Button("Genera Libro")
            with gr.Column():
                book_output = gr.Textbox(label="Libro Generato", lines=15)
        book_generate_btn.click(
            generate_book,
            inputs=[title, genre, book_style, book_content_type, book_temperature, book_max_new_tokens],
            outputs=book_output,
        )

# Porta e host configurabili (FractalNova)
SERVER_PORT = int(os.getenv("FRACTALNOVA_PORT", "7860"))
SERVER_NAME = os.getenv("FRACTALNOVA_HOST", "0.0.0.0")
demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT) 