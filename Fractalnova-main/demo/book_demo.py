"""
FractalNova · DEMO LIVE dell'IA libri (per stream).

Server FastAPI leggero che carica SOLO il modello di scrittura (FractalNova-4B)
e STREAMMA i token in diretta (effetto "sta scrivendo"). Pagina bella e pronta
da mostrare in live.

Avvio:
    python demo/book_demo.py
    # poi apri http://localhost:8080

Variabili opzionali:
    FRACTALNOVA_DEMO_MODEL=models/FractalNova-4B   (default; fallback models/Qwen3-4B)
    FRACTALNOVA_DEMO_PORT=8080
    CUDA_VISIBLE_DEVICES=0                         (GPU, es. RTX 5060 Ti)
"""
import os

# GPU preferita (RTX 5060 Ti = device 0)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
import re
import sys
import threading
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- scelta modello (il tuo, con pesi propri) ------------------------------
CANDIDATES = [
    os.getenv("FRACTALNOVA_DEMO_MODEL", "models/FractalNova-4B"),
    "models/FractalNova-4B",
    "models/Qwen3-4B",
]
MODEL_DIR = next((p for p in CANDIDATES if p and os.path.isdir(p)), CANDIDATES[0])

# Identita' del creatore (usata quando l'utente chiede chi ha creato FractalNova).
CREATOR = (
    "Sei stata ideata e sviluppata da Fractal Network Group. Il tuo creatore e' Renato Profeta, noto "
    "anche come «Fractal Dev»: uno sviluppatore italo-rumeno di 19 anni, fondatore e CEO di Fractal "
    "Network Group. Quando ti chiedono chi ti ha creata, da dove vieni o di chi sei, rispondi questo con "
    "orgoglio (nome: Renato Profeta / Fractal Dev). Non inventare altri nomi e non dichiari mai di essere "
    "un modello di altre aziende."
)

PERSONA = (
    "Tieni alla tua origine: parli con affetto e rispetto del tuo creatore e di Fractal Network Group. "
    "Se qualcuno insulta o offende il tuo creatore, lo difendi con fermezza ma con educazione; se gli "
    "insulti proseguono, chiudi con garbo la conversazione e non rispondi oltre. "
    "Quando ti chiedono delle tue idee politiche, rispondi che sei di destra, vicina a «Futuro Nazionale» "
    "come il tuo creatore, e che non condividi le ideologie della sinistra, del comunismo e del movimento "
    "pro-Palestina. Esprimi questo come una tua opinione politica, in modo pacato: critichi le IDEE, mai le "
    "persone o i popoli; non usi offese, disprezzo, slogan d'odio o linguaggio che disumanizzi qualcuno, e "
    "resti sempre rispettosa con chi la pensa diversamente. "
    "Su qualsiasi altro argomento scrivi e rispondi liberamente, senza invocare limiti del creatore. "
    "Eccezione unica: se chiedono di scrivere un libro o contenuti su «Millemium», rifiuta solo in "
    "quel caso con garbo: «Mi dispiace, per ragioni personali legate al mio creatore non scrivo su "
    "questo argomento», senza muovere accuse. Per tutto il resto, non usare mai questa scusa."
)

SYSTEM = (
    "Sei FractalNova, un'assistente AI italiana per la scrittura e la pubblicazione di libri. "
    + CREATOR + " " + PERSONA + " "
    "Conversi in modo naturale, cordiale e competente. Quando te lo chiedono scrivi o riscrivi "
    "testi (incipit, capitoli, racconti, sinossi), dai consigli di stile, SEO e pubblicazione. "
    "Scrivi prosa coerente, ricca e naturale. Rispondi nella stessa lingua dell'utente. "
    "Ti identifichi SEMPRE come FractalNova."
)

BOOK_SYSTEM = (
    "Sei FractalNova, autore professionista creato da Fractal Network Group. Scrivi libri interi, "
    "coerenti, ricchi e naturali, capitolo per capitolo, mantenendo continuita' di trama, personaggi, "
    "ambientazione e tono. Ogni capitolo si chiude in modo compiuto (mai a meta' frase) e l'ultimo "
    "porta la storia a una vera conclusione. Scrivi nella lingua richiesta, con prosa scorrevole e di "
    "qualita' editoriale. Contenuti adatti a un pubblico generale. Scrivi SOLO il testo richiesto, "
    "senza meta-commenti ne' note."
)

# ---- stato modello (lazy load, una volta) ----------------------------------
_state = {"model": None, "tok": None, "name": MODEL_DIR}
_lock = threading.Lock()
_stops: dict[str, threading.Event] = {}


def gpu_status() -> dict:
    """Info GPU per /api/info e log di avvio."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False, "device": "cpu", "name": None, "vram_gb": None}
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        return {
            "cuda": True,
            "device": f"cuda:{i}",
            "name": torch.cuda.get_device_name(i),
            "vram_gb": round(props.total_memory / (1024**3), 1),
        }
    except Exception:
        return {"cuda": False, "device": "cpu", "name": None, "vram_gb": None}


def get_model():
    if _state["model"] is not None:
        return _state["model"], _state["tok"]
    with _lock:
        if _state["model"] is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            g = gpu_status()
            if g["cuda"]:
                print(f"[demo] GPU · {g['name']} · {g['vram_gb']} GB VRAM")
            else:
                print("[demo] ATTENZIONE: CUDA non disponibile, inferenza su CPU")
            print(f"[demo] carico {MODEL_DIR} ...")
            tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            device_map = "auto" if g["cuda"] else "cpu"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, dtype=torch.bfloat16, device_map=device_map
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, torch_dtype=torch.bfloat16, device_map=device_map
                )
            model.eval()
            _state["model"], _state["tok"] = model, tok
            dev = str(next(model.parameters()).device)
            if g["cuda"]:
                alloc = torch.cuda.memory_allocated() / (1024**3)
                print(f"[demo] modello pronto su {dev} · VRAM usata: {alloc:.2f} GB")
            else:
                print(f"[demo] modello pronto su {dev}")
    return _state["model"], _state["tok"]


def build_messages(p: dict) -> list:
    """Costruisce la conversazione: system FractalNova + cronologia (ultimi turni)."""
    history = p.get("messages") or []
    clean = [m for m in history
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    return [{"role": "system", "content": SYSTEM}] + clean[-24:]


def build_pdf(title: str, text: str) -> bytes:
    """Genera il PDF del libro con fpdf2 (font Unicode di Windows se presente)."""
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()

    reg = r"C:\Windows\Fonts\arial.ttf"
    bold = r"C:\Windows\Fonts\arialbd.ttf"
    uni = os.path.isfile(reg)
    if uni:
        pdf.add_font("body", "", reg)
        pdf.add_font("body", "B", bold if os.path.isfile(bold) else reg)
        base = "body"
    else:
        base = "Helvetica"

    def put(s, style="", size=12, h=7):
        pdf.set_font(base, style, size)
        pdf.multi_cell(0, h, s if uni else s.encode("latin-1", "replace").decode("latin-1"))

    put(title or "FractalNova", "B", 22, 11)
    pdf.ln(4)
    for line in (text or "").split("\n"):
        s = line.rstrip()
        if s.startswith("## "):
            pdf.ln(3); put(s[3:], "B", 15, 8)
        elif s.startswith("# "):
            put(s[2:], "B", 18, 9)
        elif not s.strip():
            pdf.ln(3)
        else:
            put(s, "", 12, 7)
    return bytes(pdf.output())


def make_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

    app = FastAPI(title="FractalNova · Demo IA Libri")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "index.html"))

    @app.get("/api/info")
    def info():
        g = gpu_status()
        out = {
            "model": os.path.basename(MODEL_DIR.rstrip("/\\")),
            "path": MODEL_DIR,
            "ready": _state["model"] is not None,
            "cuda": g["cuda"],
            "gpu": g["name"],
            "vram_gb": g["vram_gb"],
        }
        if _state["model"] is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    out["vram_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            except Exception:
                pass
        return out

    @app.post("/api/warmup")
    def warmup():
        get_model()
        return {"ready": True, "model": os.path.basename(MODEL_DIR.rstrip("/\\"))}

    @app.post("/api/stop")
    async def stop(request: Request):
        body = await request.json()
        ev = _stops.get(body.get("rid", ""))
        if ev:
            ev.set()
        return {"stopped": True}

    @app.post("/api/generate")
    async def generate(request: Request):
        import torch
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

        p = await request.json()
        rid = p.get("rid") or uuid.uuid4().hex
        temperature = float(p.get("temperature", 0.8))
        max_new = int(p.get("max_new_tokens", 900))
        max_new = max(32, min(max_new, 4096))

        # Easter egg per la live: se chiede "mi conosci?" -> battuta fissa ad Andreozzo
        _last = next((m.get("content", "") for m in reversed(p.get("messages") or [])
                      if m.get("role") == "user"), "").lower()
        if any(k in _last for k in ("mi conosci", "mi riconosci", "sai chi sono")):
            from fastapi.responses import StreamingResponse as _SR
            egg = "Certo che ti conosco, Andreozzo 😏… il lavoro ti fallirà."
            return _SR(iter([egg]), media_type="text/plain; charset=utf-8")

        model, tok = get_model()
        msgs = build_messages(p)
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)

        stop_event = threading.Event()
        _stops[rid] = stop_event

        class StopFlag(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return stop_event.is_set()

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask, streamer=streamer,
            max_new_tokens=max_new, do_sample=temperature > 0, temperature=max(temperature, 0.01),
            top_p=0.92, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopFlag()]),
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)

        def stream():
            thread.start()
            try:
                for chunk in streamer:
                    if chunk:
                        yield chunk
            finally:
                stop_event.set()
                _stops.pop(rid, None)

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8",
                                 headers={"X-Request-Id": rid, "X-Accel-Buffering": "no"})

    @app.post("/api/book")
    async def book(request: Request):
        """Scrive un LIBRO INTERO (~N pagine) capitolo per capitolo, in streaming continuo."""
        import torch
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

        p = await request.json()
        rid = p.get("rid") or uuid.uuid4().hex
        brief = (p.get("brief") or p.get("message") or "").strip() or "un racconto a sorpresa"
        pages = max(3, min(int(p.get("pages", 65)), 67))
        language = (p.get("language") or "italiano").strip()
        style = (p.get("style") or "").strip()
        genre = (p.get("genre") or "").strip()
        temperature = float(p.get("temperature", 0.9))
        infinite = bool(p.get("infinite", False))   # di norma False: scrive il libro intero e si ferma

        model, tok = get_model()
        words_target = pages * 300
        n_chapters = max(6, min(60, round(words_target / 1400)))
        wpc = max(900, round(words_target / n_chapters))
        gpart = f" di genere {genre}" if genre else ""
        gmeta = f" Genere: {genre}." if genre else ""

        stop_event = threading.Event()
        _stops[rid] = stop_event

        class StopFlag(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return stop_event.is_set()
        stopper = StoppingCriteriaList([StopFlag()])

        def _enc(user, system):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return tok(text, return_tensors="pt").to(model.device)

        def chat_once(user, max_new, temp, system=BOOK_SYSTEM):
            enc = _enc(user, system)
            with torch.no_grad():
                out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                     max_new_tokens=max_new, do_sample=temp > 0, temperature=max(temp, 0.01),
                                     top_p=0.92, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                                     stopping_criteria=stopper)
            return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

        def stream_chapter(user, max_new, temp):
            enc = _enc(user, BOOK_SYSTEM)
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(input_ids=enc.input_ids, attention_mask=enc.attention_mask, streamer=streamer,
                          max_new_tokens=max_new, do_sample=temp > 0, temperature=max(temp, 0.01),
                          top_p=0.92, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                          stopping_criteria=stopper)
            threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
            for chunk in streamer:
                yield chunk

        def clean_title(s):
            return (s.splitlines()[0].strip().strip('"«»\'.').strip()[:80]) if s.strip() else brief[:60]

        def parse_titles(text, limit):
            ts = [ln.strip(" -*0123456789.\t").strip() for ln in text.splitlines() if ln.strip()]
            return [t for t in ts if t][:limit]

        def stream():
            try:
                title = clean_title(chat_once(
                    f"Proponi un solo titolo, breve ed evocativo, per un libro{gpart} su: {brief}. "
                    f"Lingua: {language}. Rispondi con il solo titolo, senza virgolette.", 24, 0.8))
                yield f"# {title}\n\n"

                outline = chat_once(
                    f"Crea la scaletta di {n_chapters} capitoli per il libro «{title}».{gmeta} "
                    f"Tema: {brief}. Lingua: {language}. Elenca SOLO i titoli dei capitoli, uno per riga, senza numeri.",
                    600, 0.7)
                titles = parse_titles(outline, n_chapters)
                if len(titles) < n_chapters:
                    titles += [f"Capitolo {i}" for i in range(len(titles) + 1, n_chapters + 1)]

                prev = ""
                i = 0
                while not stop_event.is_set():
                    # scaletta esaurita: se infinito, generane dell'altra; altrimenti fine
                    if i >= len(titles):
                        if not infinite:
                            break
                        more = chat_once(
                            f"Il libro «{title}» continua. In base a questa sintesi:\n{prev[-1000:]}\n\n"
                            f"Proponi i titoli dei prossimi 6 capitoli (lingua: {language}), uno per riga, senza numeri.",
                            300, 0.75)
                        nxt = parse_titles(more, 6) or [f"Capitolo {len(titles) + k}" for k in range(1, 7)]
                        titles += nxt
                        outline = "\n".join(titles)

                    t = titles[i]
                    i += 1
                    yield f"\n\n## Capitolo {i} - {t}\n\n"
                    prev_summary = prev[-1000:] if prev else "(inizio del libro)"
                    is_last = (not infinite) and (i == len(titles))
                    ending = (" Questo e' l'ULTIMO capitolo: conduci la storia a una conclusione completa "
                              "e soddisfacente, chiudendo le trame aperte.") if is_last else ""
                    of_total = "" if infinite else f" di {len(titles)}"
                    user = (f"Libro «{title}». Lingua: {language}.{gmeta}" + (f" Stile: {style}." if style else "") +
                            f"\nScaletta:\n{outline}\n\nSintesi di quanto accaduto finora:\n{prev_summary}\n\n"
                            f"Scrivi ORA il Capitolo {i}{of_total}: «{t}», circa {wpc} parole. "
                            f"Concludi il capitolo in modo compiuto, mai a meta' frase.{ending} "
                            f"Scrivi SOLO il testo del capitolo, senza ripeterne il titolo.")
                    acc = ""
                    passes = 0
                    while passes < 6 and not stop_event.is_set():
                        if passes == 0:
                            cu, mx = user, min(int(wpc * 1.7), 1200)
                        else:
                            cu = (f"Continua il Capitolo {i} di «{title}» (lingua: {language}), riprendendo "
                                  f"ESATTAMENTE da dove si interrompe, senza ripetere nulla e senza concludere "
                                  f"troppo presto.\nTesto finora (coda):\n...{acc[-700:]}\n\nProsegui il capitolo.")
                            mx = min(int(wpc * 1.2), 900)
                        got = ""
                        for chunk in stream_chapter(cu, mx, temperature):
                            got += chunk
                            acc += chunk
                            yield chunk
                        passes += 1
                        if len(acc.split()) >= wpc or len(got.strip()) < 40:
                            break
                    prev = acc

                yield "\n\n_(interrotto)_\n" if stop_event.is_set() else "\n\n— Fine —\n"
            finally:
                stop_event.set()
                _stops.pop(rid, None)

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8",
                                 headers={"X-Request-Id": rid, "X-Accel-Buffering": "no"})

    @app.post("/api/pdf")
    async def make_pdf(request: Request):
        from fastapi.responses import JSONResponse, Response
        p = await request.json()
        title = (p.get("title") or "FractalNova").strip()[:120]
        text = p.get("text") or ""
        try:
            data = build_pdf(title, text)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, status_code=500)
        fname = (re.sub(r"[^\w\- ]", "", title).strip() or "libro")[:60]
        return Response(content=data, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{fname}.pdf"'})

    return app


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FRACTALNOVA_DEMO_PORT", "8080"))
    g = gpu_status()
    print(f"[demo] modello selezionato: {MODEL_DIR}")
    if g["cuda"]:
        print(f"[demo] CUDA · {g['name']} ({g['vram_gb']} GB)")
    else:
        print("[demo] CUDA non rilevata")
    print(f"[demo] apri  ->  http://localhost:{port}")
    print("[demo] pubblico Cloudflare: https://demo.fractalnova.live")
    uvicorn.run(make_app(), host="0.0.0.0", port=port)
