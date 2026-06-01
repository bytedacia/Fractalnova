"""
FractalNova · server FLUX "cosi' com'e'" (frozen) da hostare su un server con GPU.

Espone l'endpoint che FractalNova-CALM chiama (inference/calm_cover.py --backend endpoint):
    POST /generate  {prompt, num_inference_steps, width, height, guidance_scale} -> PNG
    GET  /health

FLUX resta AS-IS (nessun training). Su un server con VRAM adeguata gira a piena
velocita'; su GPU piccola usa --offload.

Deploy:
    pip install diffusers transformers torch sentencepiece fastapi "uvicorn[standard]"
    python serving/flux_server.py --model black-forest-labs/FLUX.1-schnell --port 8500
Poi dal client:
    export FLUX_ENDPOINT_URL=http://IL_TUO_SERVER:8500/generate
    python inference/calm_cover.py --book-context "..."
"""
from __future__ import annotations

import argparse
import io
import os

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel


class GenRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 4
    width: int = 768
    height: int = 1024
    guidance_scale: float = 0.0


def build_app(model_id: str, offload: bool = False) -> FastAPI:
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")

    app = FastAPI(title="FractalNova · FLUX server (as-is)")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id}

    @app.post("/generate")
    def generate(req: GenRequest):
        image = pipe(
            prompt=req.prompt, num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale, width=req.width, height=req.height,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--offload", action="store_true", help="CPU offload (GPU piccola)")
    args = ap.parse_args()

    import uvicorn
    uvicorn.run(build_app(args.model, args.offload), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
