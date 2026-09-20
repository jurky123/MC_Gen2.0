"""Reference-image conditioning for the HD->MC Stylizer.

Two complementary paths (design doc section 4.3):
  1. spatial adapter: the 64x64 reference is encoded to a 16x16 token grid that
     is position-aligned with the 32x32/patch2 target grid; a zero-initialised
     gated residual injects it into the image tokens ("where things are").
  2. reference cross-attention: image tokens globally query the reference tokens
     through their own K/V (never merged with the text K/V), for palette/style
     transfer and robustness to misalignment.

Both are zero-initialised so a freshly initialised Stylizer behaves exactly like
the base t2i model until training moves them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class ReferenceEncoder(nn.Module):
    """(B, 4, S, S) reference -> (B, (S/4)^2, D) tokens (S=64 -> 16x16)."""

    def __init__(self, hidden_size, in_channels=4, stem=128, ref_size=64, bias=False):
        super().__init__()
        self.ref_size = ref_size
        self.grid = ref_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, stem, 3, padding=1, bias=bias),
            nn.GroupNorm(8, stem),
            nn.SiLU(),
            nn.Conv2d(stem, stem * 2, 3, stride=2, padding=1, bias=bias),
            nn.GroupNorm(8, stem * 2),
            nn.SiLU(),
            nn.Conv2d(stem * 2, hidden_size, 3, stride=2, padding=1, bias=bias),
            nn.GroupNorm(8, hidden_size),
        )

    def forward(self, ref):
        if ref.shape[-1] != self.ref_size:
            ref = F.interpolate(ref, size=(self.ref_size, self.ref_size),
                                mode="nearest")
        h = self.net(ref)
        return h.flatten(2).transpose(1, 2)  # (B, G*G, D)


class SpatialAdapter(nn.Module):
    """Zero-initialised gated per-position residual from reference tokens."""

    def __init__(self, hidden_size, bias=False):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.proj.weight)

    def forward(self, img_tokens, ref_tokens):
        if ref_tokens.shape[1] != img_tokens.shape[1]:
            raise ValueError(
                f"reference tokens ({ref_tokens.shape[1]}) and image tokens "
                f"({img_tokens.shape[1]}) must align")
        return img_tokens + torch.tanh(self.gate) * self.proj(ref_tokens)


class ReferenceCrossAttention(nn.Module):
    """Image tokens query the reference tokens with their own K/V + gate."""

    def __init__(self, hidden_size, num_heads, bias=False, qk_norm=True,
                 mlp_ratio=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=bias),
            nn.SiLU(),
            nn.Linear(inner, hidden_size, bias=bias),
        )
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, img_tokens, ref_tokens):
        B, N, D = img_tokens.shape
        M = ref_tokens.shape[1]
        h = self.num_heads
        d = self.head_dim
        q = self.q(self.norm1(img_tokens)).view(B, N, h, d)
        k = self.k(ref_tokens).view(B, M, h, d)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v(ref_tokens).view(B, M, h, d).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        h1 = img_tokens + torch.tanh(self.gate) * self.out(attn)
        return h1 + torch.tanh(self.gate) * self.mlp(self.norm2(h1))
