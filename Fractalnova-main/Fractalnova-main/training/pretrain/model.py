"""
FractalNova-Core · architettura del modello (da zero).

Decoder-only Transformer in stile moderno (come i modelli famosi, in piccolo):
  - RMSNorm (pre-norm)
  - RoPE (rotary positional embeddings)
  - Attenzione causale via F.scaled_dot_product_attention
  - MLP SwiGLU
  - weight tying tra embedding e testa di output

Tarato per essere addestrato DA ZERO su una singola GPU consumer (es. RTX 5060 Ti 16GB).
Non e' un modello di frontiera: e' il nucleo proprietario, piccolo ma interamente tuo.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 1024        # lunghezza di contesto
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768             # ~124M parametri con queste dimensioni
    dropout: float = 0.0
    bias: bool = False            # bias nei Linear/Norm
    rope_theta: float = 10000.0
    mlp_multiple_of: int = 256    # arrotonda la hidden della MLP per efficienza


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_seq: int, theta: float):
    """Restituisce cos e sin di forma (max_seq, head_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)            # (max_seq, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim). cos/sin: (T, head_dim/2)."""
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack((xr1, xr2), dim=-1).flatten(3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim deve essere pari per RoPE"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)
        hidden = config.mlp_multiple_of * math.ceil(hidden / config.mlp_multiple_of)
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class FractalNovaCore(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # init scalata per le proiezioni residue (stile GPT-2)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequenza {T} > block_size {self.config.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        # inferenza: calcola la testa solo sull'ultima posizione
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if use_fused else {}))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
