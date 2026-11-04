"""Strict configuration. No silently ignored options or machine-specific defaults."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math
import re

import yaml


def strict(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class ModelConfig:
    size: int = 16
    width: int = 256
    depth: int = 8
    heads: int = 8
    ref_size: int = 128
    text_dim: int = 768
    text_length: int = 128
    conditioning: str = "separate"
    byte_tokens: bool = True
    text_normalization: str = "none"
    reference_encoding: str = "spatial"
    multimodal_adapter: bool = False
    visual_adapter: bool = False
    visual_dim: int = 4096
    visual_grid: int = 8
    visual_layers: list = field(default_factory=lambda: [2, 5, 8])

    def __post_init__(self):
        if type(self.multimodal_adapter) is not bool or (self.multimodal_adapter and (self.conditioning != "text_only" or self.byte_tokens)):
            raise ValueError("multimodal_adapter requires pooled text_only conditioning without byte tokens")
        if type(self.visual_adapter) is not bool:
            raise ValueError("visual_adapter must be boolean")
        if self.visual_adapter and (self.multimodal_adapter or self.conditioning != 'text_only' or self.byte_tokens):
            raise ValueError('visual_adapter requires the original text_only backbone without another adapter')
        if type(self.visual_dim) is not int or self.visual_dim < 1 or type(self.visual_grid) is not int or self.visual_grid < 1:
            raise ValueError('visual dimensions must be positive integers')
        if (not isinstance(self.visual_layers, list) or not self.visual_layers
                or any(type(i) is not int or i < 0 for i in self.visual_layers)
                or len(set(self.visual_layers)) != len(self.visual_layers)):
            raise ValueError('visual_layers must be distinct nonnegative block indices')
        if self.visual_adapter and max(self.visual_layers) >= self.depth:
            raise ValueError('visual adapter layer outside backbone')
        if self.reference_encoding not in ("spatial", "dual"):
            raise ValueError("reference_encoding must be spatial or dual")
        if self.reference_encoding == "dual" and self.conditioning != "separate":
            raise ValueError("dual reference encoding requires separate conditioning")
        if any(type(v) is not int or v < 1 for v in (self.size, self.width, self.depth, self.heads,
                                                    self.ref_size, self.text_dim, self.text_length)):
            raise ValueError("model dimensions must be positive integers")
        if self.text_normalization not in ("none", "layernorm"):
            raise ValueError("text_normalization must be none or layernorm")
        if self.conditioning not in ("shared", "separate", "text_only"):
            raise ValueError("conditioning must be shared, separate or text_only")
        if type(self.byte_tokens) is not bool:
            raise ValueError("byte_tokens must be boolean")
        if self.size not in (8, 16, 32, 64):
            raise ValueError("size must be 8 (tests), 16, 32 or 64")
        if self.width < 16 or self.width % self.heads or self.width % 4:
            raise ValueError("width must be divisible by heads and 4")
        if min(self.depth, self.heads, self.text_dim, self.text_length) < 1:
            raise ValueError("model dimensions must be positive")
        if self.ref_size < self.size or self.ref_size % self.size:
            raise ValueError("reference size must be a multiple of target size")


@dataclass
class TextConfig:
    backend: str = "hf"
    model: str = ""
    revision: str = ""
    layers: list = field(default_factory=lambda: [-1])
    instruction: str = ""
    chat_template: bool = False
    dtype: str = "float32"
    cache_mb: int = 256

    def __post_init__(self):
        if type(self.cache_mb) is not int or self.cache_mb < 0:
            raise ValueError("cache_mb must be a nonnegative integer")
        if self.backend not in ("hf", "byte", "qwen3_vl_embedding"):
            raise ValueError("unsupported text backend")
        if self.backend in ("hf", "qwen3_vl_embedding") and not self.model:
            raise ValueError("hf text backend requires an explicit local model path or pinned model ID")
        if self.backend in ("hf", "qwen3_vl_embedding") and not Path(self.model).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("remote text models require a 40-character commit SHA revision (not main/tag)")
        if not self.layers or any(not isinstance(x, int) for x in self.layers):
            raise ValueError("layers must be a nonempty integer list")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("invalid text dtype")


@dataclass
class TrainConfig:
    trainable: str = "all"
    steps: int = 10000
    batch_size: int = 16
    accumulation: int = 1
    lr: float = 0.0002
    weight_decay: float = 0.01
    warmup: int = 100
    schedule_start: int = 0
    seed: int = 0
    text_dropout: float = 0.15
    ref_dropout: float = 0.3
    paired_fraction: float = 0.7
    seam_weight: float = 0.0
    precision: str = "float32"
    eval_every: int = 250
    val_samples: int = 128
    grad_clip: float = 1.0
    task_mix: dict | None = None

    def __post_init__(self):
        if self.trainable not in ("all", "attributes", "embedding_adapter", "visual_adapter"):
            raise ValueError("trainable must be all, attributes, embedding_adapter or visual_adapter")
        if self.task_mix is not None:
            if set(self.task_mix) != {"text", "reference", "joint", "unconditional"}:
                raise ValueError("task_mix requires text/reference/joint/unconditional probabilities")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in self.task_mix.values()) or abs(sum(self.task_mix.values()) - 1) > 1e-8:
                raise ValueError("task_mix probabilities must be nonnegative and sum to 1")
        if any(type(v) is not int for v in (self.steps, self.batch_size, self.accumulation, self.warmup,
                                          self.seed, self.eval_every, self.val_samples, self.schedule_start)):
            raise ValueError("training counts/seed must be integers")
        if not all(math.isfinite(x) for x in (self.lr, self.weight_decay, self.text_dropout, self.ref_dropout,
                                              self.seam_weight, self.paired_fraction, self.grad_clip)):
            raise ValueError("training settings must be finite")
        if min(self.steps, self.batch_size, self.accumulation, self.eval_every, self.val_samples) < 1:
            raise ValueError("training counts must be positive")
        if not 0 <= self.schedule_start < self.steps:
            raise ValueError("schedule_start must be in [0, steps)")
        if not 0 <= self.warmup < self.steps - self.schedule_start:
            raise ValueError("warmup must be in [0, steps - schedule_start)")
        if self.lr <= 0 or self.grad_clip <= 0 or self.seam_weight < 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer/loss settings")
        if not all(0 <= x <= 1 for x in (self.text_dropout, self.ref_dropout, self.paired_fraction)):
            raise ValueError("dropout must be in [0,1]")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("training precision must be float32 or bfloat16")


@dataclass
class Experiment:
    model: ModelConfig
    text: TextConfig
    train: TrainConfig
    task: str = "joint"
    asset_type: str = "all"
    schema_version: int = 3
    style: str = ""
    alpha_mode: str = "continuous"
    target_scope: str = "all"
    conditioning_input: str = "native"

    def __post_init__(self):
        if self.conditioning_input not in ("native", "multimodal_embedding", "visual_tokens"):
            raise ValueError("unsupported conditioning_input")
        if self.conditioning_input == 'visual_tokens' and (not self.model.visual_adapter
                or self.text.backend != 'qwen3_vl_embedding' or self.train.trainable != 'visual_adapter'):
            raise ValueError('visual token training requires Qwen and frozen-backbone visual_adapter training')
        if self.model.visual_adapter and self.conditioning_input != 'visual_tokens':
            raise ValueError('visual adapter requires the explicit spatial token protocol')
        if self.train.trainable == 'visual_adapter' and not self.model.visual_adapter:
            raise ValueError('visual_adapter training requires visual modules')
        if self.conditioning_input == "multimodal_embedding" and (not self.model.multimodal_adapter or self.text.backend != "qwen3_vl_embedding" or self.train.trainable != "embedding_adapter"):
            raise ValueError("multimodal training requires Qwen VL and frozen-backbone embedding_adapter training")
        if self.model.multimodal_adapter and self.conditioning_input != "multimodal_embedding":
            raise ValueError("multimodal adapter requires the explicit multimodal input protocol")
        if self.model.conditioning == "text_only" and self.task != "text" and self.conditioning_input not in ("multimodal_embedding", "visual_tokens"):
            raise ValueError("text_only architecture requires text task")
        if self.train.trainable == "attributes" and self.model.reference_encoding != "dual":
            raise ValueError("attribute adapter training requires dual reference encoding")
        if self.task not in ("joint", "text", "reference"):
            raise ValueError("task must be joint, text or reference")
        if self.asset_type not in ("all", "item", "tile"):
            raise ValueError("asset_type must be all, item or tile")
        if self.schema_version not in (2, 3):
            raise ValueError("unsupported experiment schema")
        if self.target_scope not in ("all", "paired"):
            raise ValueError("target_scope must be all or paired")
        if self.alpha_mode not in ("opaque", "binary", "continuous"):
            raise ValueError("alpha_mode must be opaque, binary or continuous")
        if not isinstance(self.style, str):
            raise ValueError("style must be a string")
        if self.text.backend == "byte" and not self.model.byte_tokens:
            raise ValueError("byte backend requires byte_tokens")
        mix = self.train.task_mix
        if mix and self.task == "text" and (mix["reference"] or mix["joint"]):
            raise ValueError("text experiment cannot sample reference tasks")
        if mix and self.task == "reference" and (mix["text"] or mix["joint"]):
            raise ValueError("reference experiment permits reference/unconditional only")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name, kind in (("model", ModelConfig), ("text", TextConfig), ("train", TrainConfig)):
            value[name] = strict(kind, value.get(name, {}))
        return strict(cls, value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
