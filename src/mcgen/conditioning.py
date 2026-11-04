"""Frozen external text encoder or explicit byte-token smoke baseline. Bounded frozen-feature LRU; trainable projections are never cached."""
from collections import OrderedDict
import torch
from pathlib import Path
from dataclasses import asdict
from .catalog import file_hash
from .config import fingerprint


class TextEncoder:
    def __init__(self, cfg, model_cfg, device="cpu"):
        self.embedding = None
        if cfg.backend == "qwen3_vl_embedding":
            from .qwen_embedding import QwenVLEmbedding
            self.embedding = QwenVLEmbedding(cfg, model_cfg, device)
            self.identity = self.embedding.identity
            return
        self.cfg, self.model_cfg, self.device = cfg, model_cfg, device
        identity_config = asdict(cfg)
        identity_config.pop("cache_mb", None)
        identity = {"config": identity_config, "text_length": model_cfg.text_length}
        self.trim_padding = False  # Opt-in: remove only trailing masked feature positions.
        self.cache = OrderedDict()
        self.cache_bytes = self.hits = self.misses = 0
        self.cache_limit = cfg.cache_mb * 1024 * 1024
        local = Path(cfg.model)
        if cfg.backend == "hf" and local.is_dir():
            paths = sorted(p for p in local.iterdir() if p.is_file() and
                           p.suffix in (".json", ".safetensors", ".bin", ".model", ".txt"))
            identity["local_files"] = {p.name: file_hash(p) for p in paths}
        self.identity = fingerprint(identity)
        self.model = self.tokenizer = None
        if cfg.backend == "hf":
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision or None,
                                                           trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("text tokenizer has no padding or EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModel.from_pretrained(cfg.model, revision=cfg.revision or None,
                                                   trust_remote_code=False,
                                                   torch_dtype=getattr(torch, cfg.dtype)).to(device).eval()
            self.model.requires_grad_(False)
            dim = self.model.config.hidden_size * len(cfg.layers)
            if dim != model_cfg.text_dim:
                raise ValueError(f"text feature dim {dim} != configured {model_cfg.text_dim}")

    @torch.no_grad()
    def encode(self, prompts, output_device):
        if getattr(self, "embedding", None) is not None:
            return self.embedding.encode(prompts, output_device)
        if not prompts:
            raise ValueError("empty prompt batch")
        if self.cfg.backend == "byte" or not self.cache_limit:
            return self._encode(prompts, output_device)
        found = {}
        missing = []
        for prompt in dict.fromkeys(prompts):
            if prompt in self.cache:
                self.hits += 1
                self.cache.move_to_end(prompt)
                found[prompt] = self.cache[prompt]
            else:
                self.misses += 1
                missing.append(prompt)
        if missing:
            features, masks = self._encode(missing, "cpu")
            for i, prompt in enumerate(missing):
                # Clone avoids retaining storage for the entire encoded batch.
                item = (features[i].clone(), masks[i].clone())
                found[prompt] = item
                size = sum(x.numel() * x.element_size() for x in item)
                if size <= self.cache_limit:
                    while self.cache and self.cache_bytes + size > self.cache_limit:
                        _, old = self.cache.popitem(last=False)
                        self.cache_bytes -= sum(x.numel() * x.element_size() for x in old)
                    self.cache[prompt] = item
                    self.cache_bytes += size
        length = next(iter(found.values()))[0].shape[0]
        if getattr(self, "trim_padding", False):
            length = max(1, max((int(item[1].nonzero()[-1]) + 1 if item[1].any() else 0)
                                for item in found.values()))
        return (torch.stack([found[p][0][:length] for p in prompts]).to(output_device),
                torch.stack([found[p][1][:length] for p in prompts]).to(output_device))

    def encode_multimodal(self, prompts, images, output_device):
        if getattr(self, "embedding", None) is None:
            raise ValueError("multimodal embedding requires Qwen3-VL-Embedding")
        return self.embedding.encode_multimodal(prompts, images, output_device)

    def cache_stats(self):
        if getattr(self, "embedding", None) is not None:
            return self.embedding.cache_stats()
        return {"hits": self.hits, "misses": self.misses, "bytes": self.cache_bytes,
                "limit_bytes": self.cache_limit, "entries": len(self.cache)}

    @torch.no_grad()
    def _encode(self, prompts, output_device):
        length = self.model_cfg.text_length
        if self.cfg.backend == "byte":
            # 0=padding, 1=begin, 2..257=UTF-8. Only a runnable baseline, not
            # a substitute for pretrained semantics or a silent fallback.
            ids = torch.zeros(len(prompts), length, dtype=torch.long, device=output_device)
            for i, prompt in enumerate(prompts):
                row = [1] + [x + 2 for x in prompt.encode("utf-8")[:length - 1]]
                ids[i, :len(row)] = torch.tensor(row, device=output_device)
            return ids, ids != 0
        texts = []
        for prompt in prompts:
            if self.cfg.chat_template:
                messages = ([{"role": "system", "content": self.cfg.instruction}] if self.cfg.instruction else [])
                messages.append({"role": "user", "content": prompt})
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                                            add_generation_prompt=True)
            elif self.cfg.instruction:
                prompt = self.cfg.instruction + "\n" + prompt
            texts.append(prompt)
        inputs = self.tokenizer(texts, padding="max_length", truncation=True, max_length=length, return_tensors="pt").to(self.device)
        output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = torch.cat([output.hidden_states[i] for i in self.cfg.layers], -1)
        return hidden.to(output_device), inputs.attention_mask.bool().to(output_device)
