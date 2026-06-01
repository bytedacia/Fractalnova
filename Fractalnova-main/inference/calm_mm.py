"""
FractalNova · CALM MULTIMODALE (vision -> text). Il vero multimodale fattibile su 16GB.

L'LLM (Qwen3-4B, anchor, CONGELATO) acquisisce la VISTA tramite un encoder visivo
(SigLIP, CONGELATO) collegato dallo stesso bridge cross-attention CALM. Solo il
bridge si addestra. FractalNova puo' "vedere" un'immagine e scriverci sopra.

Perche' NON FLUX qui: FLUX (generazione immagini) e' ~12B/~24GB -> non sta in 16GB
con Qwen, e non si puo' addestrare un bridge con FLUX in memoria. FLUX resta nello
stadio di orchestrazione (l'LLM scrive -> FLUX genera la copertina, in sequenza).

  anchor : Qwen3-4B            -> genera il testo
  vision : SigLIP image encoder -> patch embeddings dell'immagine
  bridge : CrossAttnBridge (da inference/calm.py) -> inietta i patch visivi nei layer dell'anchor
"""
from __future__ import annotations

import torch
import torch.nn as nn

from inference.calm import CrossAttnBridge


class MultimodalCALM(nn.Module):
    def __init__(self, anchor_path: str, vision_path: str,
                 connect_every: int = 6, n_heads: int = 8,
                 dtype=torch.bfloat16, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, SiglipVisionModel

        self.device = device if torch.cuda.is_available() else "cpu"
        self.anchor_tok = AutoTokenizer.from_pretrained(anchor_path)
        if self.anchor_tok.pad_token is None:
            self.anchor_tok.pad_token = self.anchor_tok.eos_token

        try:
            self.anchor = AutoModelForCausalLM.from_pretrained(anchor_path, dtype=dtype, device_map={"": self.device})
        except TypeError:
            self.anchor = AutoModelForCausalLM.from_pretrained(anchor_path, torch_dtype=dtype, device_map={"": self.device})

        self.processor = AutoProcessor.from_pretrained(vision_path)
        try:
            self.vision = SiglipVisionModel.from_pretrained(vision_path, dtype=dtype).to(self.device)
        except TypeError:
            self.vision = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=dtype).to(self.device)

        for p in self.anchor.parameters():
            p.requires_grad = False
        for p in self.vision.parameters():
            p.requires_grad = False
        self.anchor.eval()
        self.vision.eval()

        anchor_dim = self.anchor.config.hidden_size
        vision_dim = self.vision.config.hidden_size
        n_layers = self.anchor.config.num_hidden_layers
        self.connect_layers = list(range(connect_every - 1, n_layers, connect_every))
        self.bridges = nn.ModuleDict({
            str(i): CrossAttnBridge(anchor_dim, vision_dim, n_heads).to(self.device, dtype)
            for i in self.connect_layers
        })
        self._vis_h = None
        for i in self.connect_layers:
            self.anchor.model.layers[i].register_forward_hook(self._make_hook(str(i)))

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._vis_h is None:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = h + self.bridges[key](h, self._vis_h)
            return ((h,) + tuple(output[1:])) if is_tuple else h
        return hook

    def trainable_parameters(self):
        return [p for p in self.bridges.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode_image(self, image):
        """image: percorso o PIL.Image -> patch embeddings (1, num_patches, vision_dim)."""
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.vision(**inputs)
        return out.last_hidden_state.to(self.device)

    def forward(self, input_ids, vis_h, attention_mask=None, labels=None):
        self._vis_h = vis_h
        try:
            return self.anchor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._vis_h = None

    @torch.no_grad()
    def generate(self, prompt: str, image, max_new_tokens: int = 300, temperature: float = 0.8, top_p: float = 0.92):
        vis_h = self.encode_image(image)
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.anchor_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.anchor_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.anchor_tok(text, return_tensors="pt").to(self.device)
        self._vis_h = vis_h
        try:
            out = self.anchor.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, pad_token_id=self.anchor_tok.eos_token_id,
            )
        finally:
            self._vis_h = None
        return self.anchor_tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _smoke(anchor: str, vision: str):
    from PIL import Image
    print(f"[MM-CALM] anchor={anchor}  vision={vision}")
    model = MultimodalCALM(anchor, vision, connect_every=6)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"[MM-CALM] bridge: {n/1e6:.1f}M parametri su layer {model.connect_layers}")
    img = Image.new("RGB", (384, 384), (40, 60, 120))  # immagine fittizia per il test
    vis_h = model.encode_image(img)
    print(f"[MM-CALM] vision patch embeddings: {tuple(vis_h.shape)}")
    text = "Descrivi questa immagine e scrivi l'incipit di un racconto ispirato."
    enc = model.anchor_tok(text, return_tensors="pt").to(model.device)
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        out = model(enc.input_ids, vis_h, labels=enc.input_ids)
    print(f"[MM-CALM] forward OK, loss = {out.loss.item():.4f}")
    out.loss.backward()
    g = sum(p.grad is not None for p in model.trainable_parameters())
    print(f"[MM-CALM] backward OK, gradienti sul bridge: {g}/{len(model.trainable_parameters())}")
    print("[MM-CALM] OK: la VISTA e' collegata all'LLM; il bridge si addestra.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="models/Qwen3-4B")
    ap.add_argument("--vision", default="models/siglip-so400m")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(args.anchor, args.vision)
