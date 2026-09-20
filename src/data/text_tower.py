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
                 add_generation_prompt=True, enable_thinking=False, pad_bucket=0):
        import threading

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.pad_bucket = int(pad_bucket or 0)
        # Hook-captured states are shared mutable state; encodes may run from
        # the pipeline worker thread and from the val pass on the main
        # thread, so the whole forward+capture must be serialised.
        self._encode_lock = threading.Lock()
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
        # Only the transformer trunk is needed: running the inner backbone
        # avoids materialising (B, L, vocab) logits, which OOMs for large
        # encode batches (27+ GiB spikes on Qwen3-8B's 151k vocab).
        self.backbone = self.model.model
        self._hook_states = {}
        for layer_idx in self.layers:
            # hidden_states[k] == output of backbone.layers[k-1] (index 0 is the
            # embedding output), so hook one layer earlier.
            hooked = layer_idx - 1
            assert hooked >= 0, f"layer {layer_idx} maps to embedding output; not supported"
            def make_hook(idx):
                def hook(module, inputs, output):
                    self._hook_states[idx] = output[0] if isinstance(output, tuple) else output
                return hook
            self.backbone.layers[hooked].register_forward_hook(make_hook(layer_idx))
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
        """Return (hidden (B, L, dim), mask (B, L) bool) on ``self.device``.

        ``pad_bucket`` rounds the padded sequence length up to a multiple so
        downstream compiled models see few distinct shapes instead of one per
        batch-longest-prompt.
        """
        if isinstance(texts, str):
            texts = [texts]
        prompts = self._prompts(texts)
        pad_to = None
        if getattr(self, "pad_bucket", 0):
            tok = self.tokenizer
            longest = max(len(tok.encode(p, add_special_tokens=False)) for p in prompts)
            pad_to = min(self.max_length,
                         int(-(-longest // self.pad_bucket) * self.pad_bucket))
        if pad_to:
            enc = self.tokenizer(prompts, padding="max_length", truncation=True,
                                 max_length=pad_to, return_tensors="pt")
        else:
            enc = self.tokenizer(prompts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device).bool()
        with self._encode_lock:
            try:
                with torch.no_grad():
                    self.backbone(input_ids=input_ids, attention_mask=mask, use_cache=False)
                if self.layers:
                    if len(self._hook_states) != len(self.layers):
                        missing = [li for li in self.layers if li not in self._hook_states]
                        raise RuntimeError(f"tower hooks missing layers: {missing}")
                    hidden = torch.cat([self._hook_states[li] for li in self.layers], dim=-1)
                else:
                    hidden = self.backbone(input_ids=input_ids, attention_mask=mask,
                                           use_cache=False).last_hidden_state
            finally:
                # Never let a failed/interleaved forward leave stale states for
                # the next encode.
                self._hook_states.clear()
        return hidden, mask


def get_text_encoder(model_name, device="cuda", dtype="bfloat16", max_length=512,
                     revision=None, instruction="", layers=None, pad_bucket=0):
    key = (model_name, revision, max_length, device, tuple(layers or ()), int(pad_bucket or 0))
    encoder = _ENC_CACHE.get(key)
    if encoder is None:
        encoder = FrozenTextEncoder(model_name=model_name, device=device, dtype=dtype,
                                    max_length=max_length, revision=revision,
                                    instruction=instruction, layers=layers,
                                    pad_bucket=pad_bucket)
        _ENC_CACHE[key] = encoder
    return encoder
