"""
FractalNova · MULTI-CALM (anchor + PIU' modelli augmenting) per gli agenti chat.

Compone 3+ modelli CONGELATI: un anchor (cervello) + N augmenting, ognuno collegato
da un proprio BRIDGE cross-attention. Solo i bridge si addestrano.

Per gli agenti FractalNova:
  anchor     = mistralai/Mistral-7B-Instruct-v0.3   (coerenza)
  augmenting = nvidia/Nemotron-Mini-4B-Instruct      (roleplay/personaggi)
             + microsoft/Phi-3.5-mini-instruct       (ragionamento, lignaggio diverso)

NB MEMORIA: 3 modelli (~30GB) -> SERVER (A100/H100). Non sta nei 16GB locali.
I 3 agenti (Andreozzo/AlexsanderXXL/Matte) condividono questo CALM e si distinguono
per system persona + (opz.) adapter persona.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from inference.calm import CrossAttnBridge


class MultiCALM(nn.Module):
    def __init__(self, anchor_path: str, aug_paths: List[str],
                 connect_every: int = 6, n_heads: int = 8,
                 dtype=torch.bfloat16, device: str = "cuda", trust_remote_code: bool = True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device if torch.cuda.is_available() else "cpu"

        def load(path):
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": self.device}, trust_remote_code=trust_remote_code)
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map={"": self.device}, trust_remote_code=trust_remote_code)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            return m, tok

        self.anchor, self.anchor_tok = load(anchor_path)
        self.augs, self.aug_toks = [], []
        for ap in aug_paths:
            m, t = load(ap)
            self.augs.append(m)
            self.aug_toks.append(t)

        anchor_dim = self.anchor.config.hidden_size
        aug_dims = [m.config.hidden_size for m in self.augs]
        n_layers = self.anchor.config.num_hidden_layers
        self.connect_layers = list(range(connect_every - 1, n_layers, connect_every))

        # per ogni layer di aggancio: un bridge per ciascun augmenting
        self.bridges = nn.ModuleDict({
            str(i): nn.ModuleList([CrossAttnBridge(anchor_dim, d, n_heads).to(self.device, dtype) for d in aug_dims])
            for i in self.connect_layers
        })
        self._aug_hs: Optional[List[torch.Tensor]] = None
        for i in self.connect_layers:
            self.anchor.model.layers[i].register_forward_hook(self._make_hook(str(i)))

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._aug_hs is None:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            for bridge, aug_h in zip(self.bridges[key], self._aug_hs):
                h = h + bridge(h, aug_h)
            return ((h,) + tuple(output[1:])) if is_tuple else h
        return hook

    def trainable_parameters(self):
        return [p for p in self.bridges.parameters() if p.requires_grad]

    @torch.no_grad()
    def _aug_hidden(self, text: str, max_len: int = 512):
        hs = []
        for m, tok in zip(self.augs, self.aug_toks):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
            out = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            hs.append(out.hidden_states[-1].to(self.device))
        return hs

    def forward(self, anchor_ids, aug_texts: List[str], attention_mask=None, labels=None):
        # aug_texts: stesso testo per ogni augmenting (tokenizzato col suo tokenizer)
        self._aug_hs = []
        with torch.no_grad():
            for m, tok, text in zip(self.augs, self.aug_toks, aug_texts):
                enc = tok(text, return_tensors="pt", truncation=True, max_length=anchor_ids.shape[1] + 64).to(self.device)
                self._aug_hs.append(m(input_ids=enc.input_ids, output_hidden_states=True).hidden_states[-1].to(self.device))
        try:
            return self.anchor(input_ids=anchor_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._aug_hs = None

    @torch.no_grad()
    def generate(self, system: str, user: str, max_new_tokens: int = 400, temperature: float = 0.85, top_p: float = 0.92):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.anchor_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{system}\n\nUtente: {user}\nAssistente:"
        self._aug_hs = self._aug_hidden(text)
        enc = self.anchor_tok(text, return_tensors="pt").to(self.device)
        try:
            out = self.anchor.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, pad_token_id=self.anchor_tok.eos_token_id,
            )
        finally:
            self._aug_hs = None
        return self.anchor_tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
