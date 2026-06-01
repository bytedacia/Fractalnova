"""
FractalNova · training del bridge CALM-CON-FLUX (sul SERVER grande).

FLUX e Qwen restano FROZEN; si addestra SOLO il bridge che proietta le
rappresentazioni di FractalNova nel condizionamento di FLUX.

Due modalita':
  --mode embed  (robusta, consigliata per iniziare): il bridge impara a riprodurre
       gli embedding di condizionamento di FLUX (T5 + CLIP pooled) per delle caption.
       Cosi' "parla la lingua" di FLUX. Dati: --captions file.txt (una caption per riga).
  --mode flow   (avanzata): flow-matching su coppie (immagine, contesto): la
       comprensione di FractalNova guida la generazione vera. Dati: --pairs cartella
       con coppie  nome.png + nome.txt. SPERIMENTALE: valida sul tuo diffusers.

Uso:
    python training/train_calm_flux.py --calm-dir training/outputs/fractalnova-calm \
        --flux models/FLUX.1-schnell --mode embed --captions training/data/captions.txt --epochs 5
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.calm_flux import FractalNovaFlux  # noqa: E402


def _save(fnf, out):
    os.makedirs(out, exist_ok=True)
    torch.save(fnf.bridge.state_dict(), os.path.join(out, "flux_bridge.pt"))
    print(f"[calm-flux] bridge salvato -> {os.path.join(out, 'flux_bridge.pt')}")


def train_embed(fnf, captions, epochs, lr, out):
    """Il bridge impara a mappare FractalNova -> condizionamento FLUX (regressione embedding)."""
    opt = torch.optim.AdamW(fnf.bridge_parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(captions)
        run = 0.0
        for i, cap in enumerate(captions):
            with torch.no_grad():
                tgt_pe, tgt_ppe = fnf.flux_target_embeds(cap)   # embedding "veri" di FLUX
            pe, ppe = fnf.condition(cap)                        # dal bridge
            loss = F.mse_loss(ppe.float(), tgt_ppe.float()) + \
                F.mse_loss(pe.mean(1).float(), tgt_pe.mean(1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fnf.bridge_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            if (i + 1) % 20 == 0:
                print(f"  [embed] ep{ep+1} {i+1}/{len(captions)} loss {run/20:.4f}")
                run = 0.0
    _save(fnf, out)


def train_flow(fnf, pairs, epochs, lr, out, size=1024):
    """Flow-matching: la comprensione di FractalNova guida FLUX. SPERIMENTALE."""
    from PIL import Image
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    vae = fnf.flux.vae
    transformer = fnf.flux.transformer
    opt = torch.optim.AdamW(fnf.bridge_parameters(), lr=lr)
    sf = getattr(vae.config, "scaling_factor", 0.3611)
    shift = getattr(vae.config, "shift_factor", 0.1159)

    def load_img(p):
        img = Image.open(p).convert("RGB").resize((size, size))
        import numpy as np
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
        return x.unsqueeze(0).to(fnf.device, fnf.dtype)

    for ep in range(epochs):
        random.shuffle(pairs)
        for i, (img_p, ctx) in enumerate(pairs):
            with torch.no_grad():
                px = load_img(img_p)
                lat = vae.encode(px).latent_dist.sample()
                lat = (lat - shift) * sf
            b, c, h, w = lat.shape
            packed = FluxPipeline._pack_latents(lat, b, c, h, w)
            img_ids = FluxPipeline._prepare_latent_image_ids(b, h // 2, w // 2, fnf.device, fnf.dtype)

            noise = torch.randn_like(packed)
            t = torch.rand(b, device=fnf.device)
            tt = t.view(b, 1, 1)
            noised = (1 - tt) * packed + tt * noise
            target = noise - packed                      # velocita' (rectified flow)

            pe, ppe = fnf.condition(ctx)
            txt_ids = torch.zeros(pe.shape[1], 3, device=fnf.device, dtype=fnf.dtype)
            pred = transformer(
                hidden_states=noised, timestep=t,
                encoder_hidden_states=pe.to(fnf.dtype), pooled_projections=ppe.to(fnf.dtype),
                img_ids=img_ids, txt_ids=txt_ids, return_dict=False,
            )[0]
            loss = F.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fnf.bridge_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (i + 1) % 5 == 0:
                print(f"  [flow] ep{ep+1} {i+1}/{len(pairs)} loss {loss.item():.4f}")
    _save(fnf, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm-dir", default="training/outputs/fractalnova-calm")
    ap.add_argument("--flux", default="models/FLUX.1-schnell")
    ap.add_argument("--out", default="training/outputs/fractalnova-calm-flux")
    ap.add_argument("--mode", choices=["embed", "flow"], default="embed")
    ap.add_argument("--captions", help="[embed] file con una caption per riga")
    ap.add_argument("--pairs", help="[flow] cartella con coppie nome.png + nome.txt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    fnf = FractalNovaFlux(args.calm_dir, args.flux)
    n = sum(p.numel() for p in fnf.bridge_parameters())
    print(f"[calm-flux] bridge addestrabile: {n/1e6:.1f}M parametri | FLUX+Qwen FROZEN")

    if args.mode == "embed":
        if not args.captions or not os.path.exists(args.captions):
            raise SystemExit("--mode embed richiede --captions file.txt (una caption per riga)")
        caps = [ln.strip() for ln in open(args.captions, encoding="utf-8") if ln.strip()]
        print(f"[calm-flux] embed: {len(caps)} caption")
        train_embed(fnf, caps, args.epochs, args.lr, args.out)
    else:
        if not args.pairs:
            raise SystemExit("--mode flow richiede --pairs cartella con nome.png + nome.txt")
        pairs = []
        for png in glob.glob(os.path.join(args.pairs, "*.png")):
            txt = os.path.splitext(png)[0] + ".txt"
            if os.path.exists(txt):
                pairs.append((png, open(txt, encoding="utf-8").read().strip()))
        if not pairs:
            raise SystemExit(f"Nessuna coppia .png+.txt in {args.pairs}")
        print(f"[calm-flux] flow: {len(pairs)} coppie (immagine, contesto)")
        train_flow(fnf, pairs, args.epochs, args.lr, args.out)


if __name__ == "__main__":
    main()
