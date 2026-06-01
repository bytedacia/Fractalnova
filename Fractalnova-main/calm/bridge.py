import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BridgeOutput:
    hidden_states: torch.FloatTensor
    attention_weights: Optional[torch.FloatTensor] = None
    bridge_loss: Optional[torch.FloatTensor] = None


class CrossAttentionBridge(nn.Module):
    def __init__(
        self,
        anchor_hidden_dim: int,
        augmenting_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.anchor_hidden_dim = anchor_hidden_dim
        self.augmenting_hidden_dim = augmenting_hidden_dim
        self.num_heads = num_heads
        self.head_dim = anchor_hidden_dim // num_heads
        assert anchor_hidden_dim % num_heads == 0, "anchor_hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(anchor_hidden_dim, anchor_hidden_dim, bias=False)
        self.k_proj = nn.Linear(augmenting_hidden_dim, anchor_hidden_dim, bias=False)
        self.v_proj = nn.Linear(augmenting_hidden_dim, anchor_hidden_dim, bias=False)
        self.out_proj = nn.Linear(anchor_hidden_dim, anchor_hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(anchor_hidden_dim)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        anchor_hidden: torch.FloatTensor,
        augmenting_hidden: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> BridgeOutput:
        batch_size, seq_len_anchor, _ = anchor_hidden.shape
        _, seq_len_aug, _ = augmenting_hidden.shape

        q = self.q_proj(anchor_hidden)
        k = self.k_proj(augmenting_hidden)
        v = self.v_proj(augmenting_hidden)

        q = q.view(batch_size, seq_len_anchor, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_aug, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_aug, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_anchor, -1)
        attn_output = self.out_proj(attn_output)

        gate_val = torch.sigmoid(self.gate)
        output = anchor_hidden + gate_val * attn_output
        output = self.layer_norm(output)

        return BridgeOutput(
            hidden_states=output,
            attention_weights=attn_weights if output_attentions else None,
        )
