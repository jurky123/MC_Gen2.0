import math

import torch


def precompute_2d_freqs(head_dim, height, width, base=10000.0):
    assert head_dim % 4 == 0
    n_pairs = head_dim // 2
    inv = base ** (-2.0 * torch.arange(n_pairs, dtype=torch.float32) / head_dim)
    half = n_pairs // 2
    freq_h = inv[:half]
    freq_w = inv[half:]
    h_idx = torch.arange(height, dtype=torch.float32)
    w_idx = torch.arange(width, dtype=torch.float32)
    ang_h = h_idx[:, None] * freq_h[None, :]
    ang_w = w_idx[:, None] * freq_w[None, :]
    ang_h = ang_h[:, None, :].expand(height, width, half)
    ang_w = ang_w[None, :, :].expand(height, width, half)
    ang = torch.cat([ang_h, ang_w], dim=-1)
    ang = ang.reshape(height * width, n_pairs)
    return torch.cos(ang), torch.sin(ang)


class Rotary2D:
    def __init__(self, head_dim, base=10000.0):
        self.head_dim = head_dim
        self.base = base
        self._cache = {}

    def get(self, height, width, device, dtype):
        key = (height, width)
        if key not in self._cache:
            cos, sin = precompute_2d_freqs(self.head_dim, height, width, self.base)
            self._cache[key] = (cos, sin)
        cos, sin = self._cache[key]
        return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def pad_rope(cos, sin, text_len):
    b = cos.shape[-1]
    ones = torch.ones(text_len, b, dtype=cos.dtype, device=cos.device)
    zeros = torch.zeros(text_len, b, dtype=cos.dtype, device=cos.device)
    cos_full = torch.cat([ones, cos], dim=0)
    sin_full = torch.cat([zeros, sin], dim=0)
    return cos_full, sin_full
