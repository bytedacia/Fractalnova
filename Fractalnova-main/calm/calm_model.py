import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .config import CALMConfig
from .bridge import CrossAttentionBridge


class CALMModel(nn.Module):
    def __init__(self, config: CALMConfig):
        super().__init__()
        self.config = config

        print(f"Loading anchor model: {config.anchor_model_name}")
        self.anchor = AutoModelForCausalLM.from_pretrained(
            config.anchor_model_name,
            torch_dtype=getattr(torch, config.anchor_dtype),
            device_map=config.device_map_anchor,
            trust_remote_code=True,
        )

        print(f"Loading augmenting model: {config.augmenting_model_name}")
        self.augmenting = AutoModelForCausalLM.from_pretrained(
            config.augmenting_model_name,
            torch_dtype=getattr(torch, config.augmenting_dtype),
            device_map=config.device_map_augmenting,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.anchor_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.freeze_anchor:
            for param in self.anchor.parameters():
                param.requires_grad = False

        if config.freeze_augmenting:
            for param in self.augmenting.parameters():
                param.requires_grad = False

        anchor_config = self.anchor.config
        augmenting_config = self.augmenting.config

        anchor_hidden_dim = getattr(anchor_config, "hidden_size", anchor_config.hidden_dim)
        augmenting_hidden_dim = getattr(augmenting_config, "hidden_size", augmenting_config.hidden_dim)

        self.bridges = nn.ModuleList([
            CrossAttentionBridge(
                anchor_hidden_dim=anchor_hidden_dim,
                augmenting_hidden_dim=augmenting_hidden_dim,
                num_heads=config.bridge_num_heads,
                dropout=config.bridge_dropout,
            )
            for _ in config.bridge_layers
        ])

        self.bridge_layer_indices = config.bridge_layers
        self.total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable bridge parameters: {self.total_trainable:,}")

        self.augmenting_hidden_cache = None
        self._hooks = []
        self._register_bridge_hooks()

    def _get_transformer_layers(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer
        raise AttributeError("Could not locate transformer layers in the model")

    def _bridge_hook(self, layer_idx):
        def hook(module, args, output):
            if self.augmenting_hidden_cache is None:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            aug_idx = min(layer_idx, len(self.augmenting_hidden_cache) - 2)
            aug_hidden = self.augmenting_hidden_cache[aug_idx + 1]

            bridge_idx = self.bridge_layer_indices.index(layer_idx)
            bridge_out = self.bridges[bridge_idx](
                anchor_hidden=hidden,
                augmenting_hidden=aug_hidden,
            )

            if isinstance(output, tuple):
                return (bridge_out.hidden_states,) + output[1:]
            return bridge_out.hidden_states

        return hook

    def _register_bridge_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

        layers = self._get_transformer_layers(self.anchor)
        for idx in self.bridge_layer_indices:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._bridge_hook(idx))
                self._hooks.append(hook)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        aug_max_len = self.config.augmenting_max_length
        aug_input_ids = input_ids[:, :aug_max_len]
        aug_attention_mask = (
            attention_mask[:, :aug_max_len]
            if attention_mask is not None
            else None
        )

        with torch.no_grad():
            aug_outputs = self.augmenting(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            self.augmenting_hidden_cache = aug_outputs.hidden_states

        outputs = self.anchor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        self.augmenting_hidden_cache = None

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        self.eval()
        generated = input_ids.clone()
        past_length = input_ids.shape[1]

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"][:, -1, :]

            if do_sample:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                pad = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
