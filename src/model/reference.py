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
    """(B, 4, S, S) reference -> (B, G*G, D) tokens.

    The output grid must match the target token grid (image_size / patch_size,
    e.g. 16x16), so the convolutional stride is derived from the reference
    resolution: 64 -> stride 4, 128 -> stride 8.
    """

    def __init__(self, hidden_size, in_channels=4, stem=128, ref_size=64,
                 grid=16, bias=False):
        super().__init__()
        self.ref_size = ref_size
        stride = max(1, int(round(ref_size / grid)))
        self.grid = ref_size // stride if ref_size % stride == 0 else grid
        strides = []
        s = 1
        while s < stride:
            step = min(2, stride // s) if stride // s >= 2 else stride // s
            step = _largest_divisor_step(s, stride)
            strides.append(step)
            s *= step
        layers = [nn.Conv2d(in_channels, stem, 3, padding=1, bias=bias),
                  nn.GroupNorm(8, stem), nn.SiLU()]
        ch = stem
        for st in strides:
            out_ch = min(ch * 2, hidden_size)
            layers += [nn.Conv2d(ch, out_ch, 3, stride=st, padding=1, bias=bias),
                       nn.GroupNorm(8, out_ch), nn.SiLU()]
            ch = out_ch
        if ch != hidden_size:
            layers += [nn.Conv2d(ch, hidden_size, 1, bias=bias),
                       nn.GroupNorm(8, hidden_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, ref):
        if ref.shape[-1] != self.ref_size:
            ref = F.interpolate(ref, size=(self.ref_size, self.ref_size),
                                mode="nearest")
        h = self.net(ref)
        return h.flatten(2).transpose(1, 2)  # (B, G*G, D)


def _largest_divisor_step(so_far, target):
    """Largest stride step (<=4) such that so_far*step divides target."""
    for step in (4, 2, 1):
        if target % (so_far * step) == 0:
            return step
    return 1


class SpatialAdapter(nn.Module):
    """Zero-initialised gated per-position residual from reference tokens."""

    def __init__(self, hidden_size, bias=False):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.gate = nn.Parameter(torch.zeros(1))
        # NOTE: proj must NOT be zero-initialised together with the gate, or
        # both stay at zero forever (dL/dgate involves proj(ref)=0 and
        # dL/dproj involves tanh(gate)=0). Output is still exactly zero at
        # init because tanh(0)=0, but the gate receives gradient.

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
        # Same zero-init trap as the spatial adapter: keep the gate at zero for
        # an identity initialisation, but leave out/mlp randomly initialised so
        # the gate gradient is non-zero.

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
