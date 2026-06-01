from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CALMConfig:
    anchor_model_name: str = "Qwen/Qwen3-4B"
    augmenting_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    bridge_layers: list[int] = field(default_factory=lambda: [8, 16, 24])
    bridge_num_heads: int = 8
    bridge_hidden_dim: int = 1024
    bridge_dropout: float = 0.1

    anchor_max_length: int = 2048
    augmenting_max_length: int = 1024

    anchor_dtype: str = "bfloat16"
    augmenting_dtype: str = "bfloat16"

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    max_steps: int = 5000
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "calm_output"
    logging_dir: str = "calm_logs"

    freeze_anchor: bool = True
    freeze_augmenting: bool = True

    device_map_anchor: str = "auto"
    device_map_augmenting: str = "auto"

    def __post_init__(self):
        assert len(self.bridge_layers) > 0, "At least one bridge layer required"
        assert 0 <= min(self.bridge_layers) and max(self.bridge_layers) < 100, "Bridge layers indices must be reasonable"
