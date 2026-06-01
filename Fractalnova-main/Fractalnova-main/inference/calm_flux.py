"""
FractalNova · CALM CON FLUX (composizione text -> image, vera).

Un BRIDGE addestrabile proietta le rappresentazioni di FractalNova-CALM (Qwen)
nello spazio di condizionamento di FLUX (embedding T5 [seq,4096] + CLIP pooled [768]).
FLUX resta FROZEN (as-is). Solo il bridge si addestra: la *comprensione* di
FractalNova guida la generazione dell'immagine, non un semplice prompt di testo.

  anchor : FractalNova-CALM (Qwen, frozen) -> hidden states del contesto libro
  bridge : proiettori addestrabili -> (prompt_embeds T5, pooled CLIP)  [QUESTO si addestra]
  FLUX   : frozen, genera l'immagine dai prompt_embeds del bridge

NB MEMORIA: FLUX (~12B) + Qwen vanno in VRAM insieme -> SERVER con GPU adeguata
(non i 16GB locali). Questo modulo e' l'architettura; training/uso sul server.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn


class FractalNovaFluxBridge(nn.Module):
    """Proietta hidden states dell'LLM -> condizionamento FLUX (T5 seq + CLIP pooled)."""

    def __init__(self, anchor_dim: int, t5_dim: int = 4096, pooled_dim: int = 768):
        super().__init__()
        self.norm = nn.LayerNorm(anchor_dim)
        self.to_t5 = nn.Sequential(
            nn.Linear(anchor_dim, t5_dim), nn.GELU(), nn.Linear(t5_dim, t5_dim),
        )
        self.to_pooled = nn.Sequential(
            nn.Linear(anchor_dim, pooled_dim), nn.GELU(), nn.Linear(pooled_dim, pooled_dim),
        )

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None):
        h = self.norm(hidden)
        prompt_embeds = self.to_t5(h)  # (B, S, 4096)
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).to(h.dtype)
            pooled_src = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled_src = h.mean(1)
        pooled = self.to_pooled(pooled_src)  # (B, 768)
        return prompt_embeds, pooled


class FractalNovaFlux(nn.Module):
    """FractalNova-CALM (frozen) + bridge (trainable) + FLUX (frozen)."""

    def __init__(self, calm_dir: str, flux_path: str, device: str = "cuda", dtype=torch.bfloat16):
        super().__init__()
        from diffusers import FluxPipeline

        from inference.calm import FractalNovaCALM

        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        with open(os.path.join(calm_dir, "calm_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.calm = FractalNovaCALM(cfg["anchor"], cfg["aug"], connect_every=cfg["connect_every"])
        self.calm.bridges.load_state_dict(torch.load(os.path.join(calm_dir, "bridge.pt"), map_location=self.device))
        self.tok = self.calm.anchor_tok
        for p in self.calm.parameters():
            p.requires_grad = False

        self.flux = FluxPipeline.from_pretrained(flux_path, torch_dtype=dtype)
        if torch.cuda.is_available():
            self.flux = self.flux.to(self.device)
        for comp in (self.flux.transformer, self.flux.vae, self.flux.text_encoder, self.flux.text_encoder_2):
            for p in comp.parameters():
                p.requires_grad = False

        anchor_dim = self.calm.anchor.config.hidden_size
        t5_dim = getattr(self.flux.text_encoder_2.config, "d_model", 4096)
        pooled_dim = getattr(self.flux.text_encoder.config, "projection_dim", 768)
        self.bridge = FractalNovaFluxBridge(anchor_dim, t5_dim, pooled_dim).to(self.device, dtype)

    def bridge_parameters(self):
        return list(self.bridge.parameters())

    @torch.no_grad()
    def context_hidden(self, text: str, max_len: int = 256):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        out = self.calm.anchor(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
        return out.hidden_states[-1], enc.attention_mask

    @torch.no_grad()
    def flux_target_embeds(self, caption: str):
        """Embedding di condizionamento 'veri' di FLUX per una caption (target di training)."""
        pe, ppe, _ = self.flux.encode_prompt(prompt=caption, prompt_2=caption,
                                             device=self.device, num_images_per_prompt=1)
        return pe, ppe

    def condition(self, text: str):
        """FractalNova -> condizionamento FLUX (prompt_embeds, pooled)."""
        h, m = self.context_hidden(text)
        return self.bridge(h, m)

    @torch.no_grad()
    def generate(self, book_context: str, steps: int = 4, width: int = 768, height: int = 1024,
                 out: str = "book_covers/calm_flux.png") -> str:
        pe, ppe = self.condition(book_context)
        image = self.flux(
            prompt_embeds=pe.to(self.flux.dtype),
            pooled_prompt_embeds=ppe.to(self.flux.dtype),
            num_inference_steps=steps, guidance_scale=0.0, width=width, height=height,
        ).images[0]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        image.save(out)
        return out
