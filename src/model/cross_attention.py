"""Cross-attention text conditioning (Q=image tokens, K/V=text tokens).

The original MMDiT-style joint attention mixes text into the image/text streams.
For long token sequences (e.g. 128 T5 tokens) cross-attention is cheaper and is
the standard SD / PixArt way to inject text: image tokens query the text tokens.

The output projection is zero-initialised so a block starts as identity and text
is learned gradually (AdaLN-Zero style), which keeps Stage 1 image weights usable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulation import Modulation, SwiGLUMLP
from .norm import RMSNorm, QKNorm


def _heads(x, heads):
    b, length, _ = x.shape
    return x.view(b, length, heads, x.shape[-1] // heads).transpose(1, 2)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, text_dim, bias=False, qk_norm=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(text_dim, 2 * dim, bias=bias)
        self.qknorm_q = QKNorm(dim)
        self.qknorm_k = QKNorm(dim)
        self.out = nn.Linear(dim, dim, bias=bias)
        nn.init.zeros_(self.out.weight)
        if bias:
            nn.init.zeros_(self.out.bias)

    def forward(self, x, text, text_mask=None):
        """x: (B, L, dim) image tokens; text: (B, Lt, text_dim); mask: (B, Lt) bool."""
        b, length, _ = x.shape
        h = self.norm(x)
        q = _heads(self.qknorm_q(self.q(h)), self.heads)
        kv = self.kv(text)
        k = _heads(self.qknorm_k(kv[..., : self.dim]), self.heads)
        v = _heads(kv[..., self.dim:], self.heads)
        attn_mask = None
        if text_mask is not None:
            # SDPA bool mask: True = participate. (B,1,1,Lt) broadcasts.
            attn_mask = text_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(b, length, self.dim)
        return x + self.out(a)


class CrossAttnBlock(nn.Module):
    """Cross-attention over text followed by an AdaLN-modulated MLP."""

    def __init__(self, dim, heads, text_dim, mlp_ratio=3.0, bias=False, qk_norm=True):
        super().__init__()
        self.cross = CrossAttention(dim, heads, text_dim, bias=bias, qk_norm=qk_norm)
        self.norm = RMSNorm(dim)
        self.mod = Modulation(dim, bias=bias)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, bias=bias)

    def forward(self, x, text, text_mask, c):
        x = self.cross(x, text, text_mask)
        shift, scale, gate, _s2, _sc2, _g2 = self.mod(c)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        res = x
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        return res + gate * self.mlp(h)
