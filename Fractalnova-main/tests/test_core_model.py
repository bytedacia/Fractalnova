"""Test dell'architettura FractalNova-Core (saltato se torch non e' installato)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")  # salta con grazia senza torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _tiny_model():
    from training.pretrain.model import FractalNovaCore, GPTConfig
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    return FractalNovaCore(cfg), cfg


def test_forward_produces_loss_and_logits():
    model, cfg = _tiny_model()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == cfg.vocab_size
    assert loss is not None and loss.item() > 0


def test_generate_extends_sequence():
    model, _ = _tiny_model()
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=8, temperature=1.0, top_k=10)
    assert out.shape[1] == 12  # 4 + 8


def test_param_count_is_reasonable():
    model, _ = _tiny_model()
    assert model.num_params() > 0


def test_weight_tying():
    model, _ = _tiny_model()
    # embedding e testa di output condividono i pesi (come nei modelli moderni)
    assert model.lm_head.weight is model.tok_emb.weight
