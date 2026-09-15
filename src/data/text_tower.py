"""Frozen token-level text encoder for on-the-fly conditioning.

Used by the cross-attention text path: the dataloader yields prompt strings and
the training loop (or the sampler) encodes them each step, so no text embedding
has to be precomputed.

Defaults to the project's frozen ``Qwen3-VL-Embedding-8B`` (a Qwen3-VL model
whose text hidden states are exposed through ``last_hidden_state``). Any HF
encoder exposing ``last_hidden_state`` works.
"""
from __future__ import annotations

import torch

_ENC_CACHE = {}


class FrozenTextEncoder:
    def __init__(self, model_name, device="cuda", dtype="bfloat16", max_length=128,
                 revision=None, trust_remote_code=True, instruction=""):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.instruction = instruction or ""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision or None, trust_remote_code=trust_remote_code
        )
        torch_dtype = getattr(torch, dtype, torch.float32) if isinstance(dtype, str) else dtype
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision or None, trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        self.dim = int(getattr(text_cfg, "hidden_size", None)
                       or getattr(cfg, "hidden_size", None)
                       or getattr(cfg, "d_model", 0))

    def _prompts(self, texts):
        tok = self.tokenizer
        template = getattr(tok, "chat_template", None)
        if not template:
            return list(texts)
        messages_extra = []
        if self.instruction:
            messages_extra.append({"role": "system", "content": [{"type": "text", "text": self.instruction}]})
        out = []
        for text in texts:
            content = [{"type": "text", "text": text}]
            out.append(tok.apply_chat_template(
                messages_extra + [{"role": "user", "content": content}],
                add_generation_prompt=False, tokenize=False,
            ))
        return out

    def encode(self, texts):
        """Return (hidden (B, L, dim), mask (B, L) bool) on ``self.device``."""
        if isinstance(texts, str):
            texts = [texts]
        prompts = self._prompts(texts)
        enc = self.tokenizer(prompts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device).bool()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        return hidden, mask


def get_text_encoder(model_name, device="cuda", dtype="bfloat16", max_length=128,
                     revision=None, instruction=""):
    key = (model_name, revision, max_length, device)
    encoder = _ENC_CACHE.get(key)
    if encoder is None:
        encoder = FrozenTextEncoder(model_name, device=device, dtype=dtype,
                                    max_length=max_length, revision=revision,
                                    instruction=instruction)
        _ENC_CACHE[key] = encoder
    return encoder
