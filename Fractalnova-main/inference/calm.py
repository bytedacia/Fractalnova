"""
FractalNova · CALM (Composition to Augment Language Models).

Compone DUE modelli CONGELATI con un BRIDGE addestrabile (cross-attention):
  - anchor     : il cervello coerente che genera (es. Qwen3-4B)
  - augmenting : un secondo modello che fornisce rappresentazioni extra
  - bridge     : cross-attention GATED (init a 0) che inietta le rappresentazioni
                 dell'augmenting in alcuni layer dell'anchor. SOLO il bridge si addestra.

NON fonde i pesi (impossibile tra architetture diverse): collega le RAPPRESENTAZIONI.
Per questo funziona anche tra famiglie diverse (Qwen + Mistral/Gemma/Llama...).

Il risultato e' un modello NUOVO: architettura composta + pesi-bridge tuoi.

Memoria (16GB): anchor 4B bf16 (~8GB) + augmenting <=2B bf16. Seq corte + batch 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CrossAttnBridge(nn.Module):
    """Cross-attention gated: Q dall'anchor, K/V dall'augmenting. Residuo gated (init 0)."""

    def __init__(self, anchor_dim: int, aug_dim: int, n_heads: int = 8):
        super().__init__()
        assert anchor_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = anchor_dim // n_heads
        self.norm_a = nn.LayerNorm(anchor_dim)
        self.q = nn.Linear(anchor_dim, anchor_dim, bias=False)
        self.k = nn.Linear(aug_dim, anchor_dim, bias=False)
        self.v = nn.Linear(aug_dim, anchor_dim, bias=False)
        self.o = nn.Linear(anchor_dim, anchor_dim, bias=False)
        # gate a 0 -> all'inizio il bridge NON disturba l'anchor (training stabile)
        self.gate = nn.Parameter(torch.zeros(1))

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, anchor_h: torch.Tensor, aug_h: torch.Tensor) -> torch.Tensor:
        a = self.norm_a(anchor_h)
        q, k, v = self._split(self.q(a)), self._split(self.k(aug_h)), self._split(self.v(aug_h))
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, Ta, hd)
        out = attn.transpose(1, 2).reshape(anchor_h.shape)
        return torch.tanh(self.gate) * self.o(out)


class FractalNovaCALM(nn.Module):
    def __init__(self, anchor_path: str, aug_path: str,
                 connect_every: int = 6, n_heads: int = 8, dtype=torch.bfloat16,
                 device: str = "cuda", trust_remote_code: bool = False):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.anchor_tok = AutoTokenizer.from_pretrained(anchor_path)
        self.aug_tok = AutoTokenizer.from_pretrained(aug_path)
        for tk in (self.anchor_tok, self.aug_tok):
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token

        self.anchor = self._load(anchor_path, dtype, trust_remote_code)
        self.aug = self._load(aug_path, dtype, trust_remote_code)
        for p in self.anchor.parameters():
            p.requires_grad = False
        for p in self.aug.parameters():
            p.requires_grad = False
        self.anchor.eval()
        self.aug.eval()

        anchor_dim = self.anchor.config.hidden_size
        aug_dim = self.aug.config.hidden_size
        n_layers = self.anchor.config.num_hidden_layers
        self.connect_layers = list(range(connect_every - 1, n_layers, connect_every))

        self.bridges = nn.ModuleDict({
            str(i): CrossAttnBridge(anchor_dim, aug_dim, n_heads).to(self.device, dtype)
            for i in self.connect_layers
        })
        self._aug_h = None  # rappresentazioni augmenting per il forward corrente
        self._register_hooks()

    def _load(self, path, dtype, trc):
        try:
            return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": self.device}, trust_remote_code=trc)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map={"": self.device}, trust_remote_code=trc)

    def _anchor_layers(self):
        # Qwen3/Llama/Mistral: <model>.model.layers
        return self.anchor.model.layers

    def _register_hooks(self):
        layers = self._anchor_layers()
        for i in self.connect_layers:
            layers[i].register_forward_hook(self._make_hook(str(i)))

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._aug_h is None:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = h + self.bridges[key](h, self._aug_h)
            if is_tuple:
                return (h,) + tuple(output[1:])
            return h
        return hook

    def forward(self, anchor_ids, aug_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            aug_out = self.aug.model(input_ids=aug_ids, output_hidden_states=True)
            self._aug_h = aug_out.hidden_states[-1].to(self.device)
        try:
            out = self.anchor(input_ids=anchor_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._aug_h = None
        return out

    def trainable_parameters(self):
        return [p for p in self.bridges.parameters() if p.requires_grad]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 400, temperature: float = 0.8, top_p: float = 0.92):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.anchor_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.anchor_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a = self.anchor_tok(text, return_tensors="pt").to(self.device)
        g = self.aug_tok(text, return_tensors="pt").to(self.device)
        # le rappresentazioni augmenting del prompt restano fisse durante la generazione
        aug_out = self.aug.model(input_ids=g.input_ids, output_hidden_states=True)
        self._aug_h = aug_out.hidden_states[-1].to(self.device)
        try:
            out = self.anchor.generate(
                input_ids=a.input_ids, attention_mask=a.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=self.anchor_tok.eos_token_id,
            )
        finally:
            self._aug_h = None
        return self.anchor_tok.decode(out[0, a.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _smoke(anchor: str, aug: str):
    print(f"[CALM] anchor={anchor}  augmenting={aug}")
    model = FractalNovaCALM(anchor, aug, connect_every=6)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"[CALM] bridge addestrabile: {n/1e6:.1f}M parametri su {len(model.connect_layers)} layer {model.connect_layers}")
    text = "Scrivi una frase d'apertura per un romanzo."
    a_ids = model.anchor_tok(text, return_tensors="pt").input_ids.to(model.device)
    g_ids = model.aug_tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        out = model(a_ids, g_ids, labels=a_ids)
    print(f"[CALM] forward OK, loss = {out.loss.item():.4f}")
    out.loss.backward()
    g = [p.grad is not None for p in model.trainable_parameters()]
    print(f"[CALM] backward OK, gradienti sul bridge: {sum(g)}/{len(g)} tensori")
    print("[CALM] OK: il bridge si addestra, anchor+augmenting restano congelati.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="models/Qwen3-4B")
    ap.add_argument("--aug", required=True, help="path/repo del modello augmenting (<=2B su 16GB)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(args.anchor, args.aug)
