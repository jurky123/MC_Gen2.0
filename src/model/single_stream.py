import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulation import Modulation, SwiGLUMLP
from .norm import RMSNorm, QKNorm
from .rope2d import apply_rotary_emb


class SingleStreamBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=3.0, bias=False, qk_norm=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.norm = RMSNorm(dim)
        self.mod = Modulation(dim, bias=bias)
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.qknorm_q = QKNorm(dim)
        self.qknorm_k = QKNorm(dim)
        self.out = nn.Linear(dim, dim, bias=bias)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, bias=bias)

    def forward(self, x, c, cos, sin):
        shift, scale, gate, shift2, scale2, gate2 = self.mod(c)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        res = x
        h = self.norm(x)
        h = h * (1.0 + scale) + shift

        qkv_h = self.qkv(h)
        B, L, _ = qkv_h.shape
        q = qkv_h[:, :, : self.dim]
        k = qkv_h[:, :, self.dim : 2 * self.dim]
        v = qkv_h[:, :, 2 * self.dim :]
        q = self.qknorm_q(q)
        k = self.qknorm_k(k)
        hd = self.head_dim
        q = q.view(B, L, self.heads, hd).transpose(1, 2)
        k = k.view(B, L, self.heads, hd).transpose(1, 2)
        v = v.view(B, L, self.heads, hd).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, self.dim)
        h = res + gate * self.out(attn)

        res2 = h
        h = self.norm(h)
        h = h * (1.0 + scale2) + shift2
        h = self.mlp(h)
        return res2 + gate2 * h
