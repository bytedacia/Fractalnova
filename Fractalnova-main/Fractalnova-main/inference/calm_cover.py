"""
FractalNova · copertina con FLUX "cosi' com'e'" (frozen), pilotato da FractalNova-CALM.

FLUX e' grande (~12B): lo si usa AS-IS in due modi (configurabili) -> nessun
download locale obbligatorio, perche' di solito FLUX vive su un SERVER:

  --backend endpoint  : chiama un server FLUX gia' hostato (HTTP). DEFAULT.
                        (vedi serving/flux_server.py per hostarlo as-is)
  --backend local     : FluxPipeline locale (server con VRAM adeguata;
                        usa --offload per stare in 16GB, piu' lento)

Flusso:
  1) FractalNova-CALM (testo) deriva il PROMPT VISIVO dal contesto del libro
  2) FLUX (as-is, frozen) genera la copertina (endpoint o locale)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calm_concept(calm_dir: str, book_context: str) -> str:
    """FractalNova-CALM deriva un prompt visivo (in inglese) dal contesto del libro."""
    from inference.calm import FractalNovaCALM
    with open(os.path.join(calm_dir, "calm_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = FractalNovaCALM(cfg["anchor"], cfg["aug"], connect_every=cfg["connect_every"])
    model.bridges.load_state_dict(torch.load(os.path.join(calm_dir, "bridge.pt"), map_location=model.device))
    prompt = model.generate(
        "From this book context, write ONE single-line English image prompt for a book cover "
        "(subject, mood, style, colors, composition). Only the prompt:\n\n" + book_context,
        max_new_tokens=80, temperature=0.7,
    ).strip().splitlines()[0]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt


def flux_endpoint(url: str, prompt: str, out: str, steps: int, width: int, height: int, api_key=None) -> str:
    """Chiama un server FLUX gia' hostato. Accetta image/* binario o JSON con base64."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"prompt": prompt, "num_inference_steps": steps, "width": width, "height": height}
    with httpx.Client(timeout=600) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if r.headers.get("content-type", "").startswith("image/"):
            data = r.content
        else:
            j = r.json()
            b64 = j.get("image") or (j.get("images") or [None])[0] or j.get("b64_json")
            if not b64:
                raise SystemExit(f"Risposta endpoint non riconosciuta: {list(j)[:6]}")
            data = base64.b64decode(str(b64).split(",")[-1])
        with open(out, "wb") as f:
            f.write(data)
    return out


def flux_local(flux_path: str, prompt: str, out: str, steps: int, width: int, height: int, offload: bool) -> str:
    """FLUX as-is in locale (frozen). Su server con VRAM: full speed; su 16GB: --offload."""
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                 height=height, width=width).images[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    image.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm-dir", default="training/outputs/fractalnova-calm")
    ap.add_argument("--book-context", required=True)
    ap.add_argument("--out", default="book_covers/calm_cover.png")
    ap.add_argument("--backend", choices=["endpoint", "local"], default=os.getenv("FLUX_BACKEND", "endpoint"))
    ap.add_argument("--flux-url", default=os.getenv("FLUX_ENDPOINT_URL"))
    ap.add_argument("--flux-key", default=os.getenv("FLUX_ENDPOINT_KEY"))
    ap.add_argument("--flux", default=os.getenv("FLUX_LOCAL_PATH", "models/FLUX.1-schnell"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--offload", action="store_true", help="CPU offload per FLUX locale su 16GB")
    args = ap.parse_args()

    print("[cover] 1/2 FractalNova-CALM deriva il concept di copertina...")
    prompt = calm_concept(args.calm_dir, args.book_context)
    print("[cover] prompt visivo:", prompt)

    print(f"[cover] 2/2 FLUX as-is ({args.backend}) genera...")
    if args.backend == "endpoint":
        if not args.flux_url:
            raise SystemExit("Backend endpoint: imposta --flux-url (o FLUX_ENDPOINT_URL) col tuo server FLUX.")
        path = flux_endpoint(args.flux_url, prompt, args.out, args.steps, args.width, args.height, args.flux_key)
    else:
        path = flux_local(args.flux, prompt, args.out, args.steps, args.width, args.height, args.offload)
    print("[cover] copertina salvata ->", path)


if __name__ == "__main__":
    main()
