from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    name: str = "MCFlowDiTBase"
    image_size: int = 32
    in_channels: int = 3
    patch_size: int = 2

    hidden_size: int = 512
    num_heads: int = 8
    mlp_ratio: float = 3.0
    text_mlp_ratio: float = 1.0

    double_stream_blocks: int = 3
    single_stream_blocks: int = 6

    text_dim: int = 768
    max_text_tokens: int = 64

    qk_norm: bool = True
    rope: str = "2d"
    activation: str = "swiglu"
    bias: bool = False

    cond_dropout: float = 0.12

    @property
    def patch_dim(self):
        return self.in_channels * self.patch_size * self.patch_size

    @classmethod
    def from_dict(cls, d):
        model = dict(d.get("model", d))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in model.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        return cls.from_dict(load_yaml(path))

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    precision: str = "bf16"
    compile: bool = False

    seed: int = 0
    steps: int = 100000
    log_every: int = 25
    save_every: int = 5000
    val_every: int = 1000
    eval_every_epochs: int = 10
    output_dir: str = "checkpoints"

    optimizer: dict = field(default_factory=lambda: {
        "name": "adamw",
        "fused": True,
        "lr": 3.0e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.03,
    })
    scheduler: dict = field(default_factory=lambda: {"type": "cosine", "warmup_steps": 1000})
    grad_clip: float = 1.0
    activation_checkpointing: bool = False
    gradient_accumulation: int = 1

    batch: dict = field(default_factory=lambda: {
        "auto_probe": True,
        "preferred_micro_batch": 128,
        "max_vram_gb": 7.2,
    })

    ema: dict = field(default_factory=lambda: {
        "enabled": True,
        "device": "cpu",
        "decay": 0.999,
        "update_every": 1,
    })

    flow: dict = field(default_factory=lambda: {"timestep_sampling": "uniform", "loss": "mse"})
    tile_loss: dict = field(default_factory=lambda: {
        "enabled": True,
        "weight": 0.03,
        "max_t": 0.7,
        "border_width": 2,
    })

    dataset: str = "configs/data/stage_a.yaml"
    text_mmap: str = ""
    log_file: str = ""

    @classmethod
    def from_dict(cls, d):
        train = dict(d.get("train", d))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in train.items() if k in known})

    @classmethod
    def from_yaml(cls, path):
        return cls.from_dict(load_yaml(path))

    def to_dict(self):
        return asdict(self)
