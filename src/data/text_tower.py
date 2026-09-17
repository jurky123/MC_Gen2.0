"""Frozen token-level text encoder for on-the-fly conditioning.

Mirrors FLUX.2 [klein]: a Qwen3 causal LM whose token hidden states from several
intermediate layers are concatenated into a long conditioning sequence
(layers 9/18/27 for Qwen3-8B, chat template with ``add_generation_prompt=True``
and ``enable_thinking=False``).

The dataloader yields prompt strings; the training loop / sampler encodes them
each step, so nothing is precomputed. The encoder is always frozen.
"""
from __future__ import annotations

import torch

_ENC_CACHE = {}


class FrozenTextEncoder:
    def __init__(self, model_name, device="cuda", dtype="bfloat16", max_length=512,
                 revision=None, trust_remote_code=True, instruction="", layers=None,
                 add_generation_prompt=True, enable_thinking=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.instruction = instruction or ""
        self.layers = [int(x) for x in (layers or [])]
        self.add_generation_prompt = bool(add_generation_prompt)
        self.enable_thinking = bool(enable_thinking)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision or None, trust_remote_code=trust_remote_code
        )
        torch_dtype = getattr(torch, dtype, torch.float32) if isinstance(dtype, str) else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision or None, trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        base_dim = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", 0))
        self.base_dim = base_dim
        self.dim = base_dim * (len(self.layers) if self.layers else 1)

    def _prompts(self, texts):
        tok = self.tokenizer
        if not getattr(tok, "chat_template", None):
            return list(texts)
        out = []
        for text in texts:
            messages = []
            if self.instruction:
                messages.append({"role": "system", "content": self.instruction})
            messages.append({"role": "user", "content": text})
            kwargs = {"tokenize": False, "add_generation_prompt": self.add_generation_prompt}
            try:
                rendered = tok.apply_chat_template(messages, enable_thinking=self.enable_thinking, **kwargs)
            except TypeError:
                rendered = tok.apply_chat_template(messages, **kwargs)
            out.append(rendered)
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
            out = self.model(input_ids=input_ids, attention_mask=mask,
                             output_hidden_states=True, use_cache=False)
        if self.layers:
            hidden_states = out.hidden_states
            stacked = torch.stack([hidden_states[i] for i in self.layers], dim=1)  # (B, C, L, d)
            hidden = stacked.permute(0, 2, 1, 3).reshape(input_ids.shape[0], input_ids.shape[1], -1)
        else:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1]
        return hidden, mask


def get_text_encoder(model_name, device="cuda", dtype="bfloat16", max_length=512,
                     revision=None, instruction="", layers=None):
    key = (model_name, revision, max_length, device, tuple(layers or ()))
    encoder = _ENC_CACHE.get(key)
    if encoder is None:
        encoder = FrozenTextEncoder(model_name, device=device, dtype=dtype,
                                    max_length=max_length, revision=revision,
                                    instruction=instruction, layers=layers)
        _ENC_CACHE[key] = encoder
    return encoder
