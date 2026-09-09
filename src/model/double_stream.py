import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulation import Modulation, SwiGLUMLP
from .norm import RMSNorm, QKNorm
from .rope2d import apply_rotary_emb


def _heads(qkv, heads):
    B, L, _ = qkv.shape
    h, d = heads, qkv.shape[-1] // heads
    return qkv.view(B, L, h, d).transpose(1, 2)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_ratio=3.0,
        text_mlp_ratio=1.0,
        bias=False,
        qk_norm=True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.img_norm = RMSNorm(dim)
        self.img_mod = Modulation(dim, bias=bias)
        self.img_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.img_qknorm_q = QKNorm(dim)
        self.img_qknorm_k = QKNorm(dim)
        self.img_out = nn.Linear(dim, dim, bias=bias)
        self.img_mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, bias=bias)

        self.txt_norm = RMSNorm(dim)
        self.txt_mod = Modulation(dim, bias=bias)
        self.txt_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.txt_qknorm_q = QKNorm(dim)
        self.txt_qknorm_k = QKNorm(dim)
        self.txt_out = nn.Linear(dim, dim, bias=bias)
        self.txt_mlp = SwiGLUMLP(dim, mlp_ratio=text_mlp_ratio, bias=bias)

    def _branch(
        self,
        x,
        kv,
        c,
        norm,
        mod,
        qkv,
        qnorm_q,
        qnorm_k,
        out_proj,
        mlp,
        cos_q,
        sin_q,
        cos_kv,
        sin_kv,
    ):
        shift, scale, gate, shift2, scale2, gate2 = mod(c)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        res = x
        h = norm(x)
        h = h * (1.0 + scale) + shift
        h = h + 0.0

        qkv_h = qkv(h)
        q = qkv_h[:, :, : self.dim]
        q = qnorm_q(q)
        q = _heads(q, self.heads)
        if cos_q is not None:
            q = apply_rotary_emb(q, cos_q, sin_q)

        qkv_kv = qkv(kv)
        k = qnorm_k(qkv_kv[:, :, : self.dim])
        v = qkv_kv[:, :, 2 * self.dim :]
        k = _heads(k, self.heads)
        v = _heads(v, self.heads)
        if cos_kv is not None:
            k = apply_rotary_emb(k, cos_kv, sin_kv)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(attn.shape[0], -1, self.dim)
        h = out_proj(attn)
        h = res + gate * h

        res2 = h
        h = norm(h)
        h = h * (1.0 + scale2) + shift2
        h = mlp(h)
        return res2 + gate2 * h

    def forward(self, img, txt, c, cos_img, sin_img, cos_kv, sin_kv):
        kv = torch.cat([txt, img], dim=1)

        img_branch_in = img
        txt_branch_in = txt
        img_out = self._branch(
            img_branch_in,
            kv,
            c,
            self.img_norm,
            self.img_mod,
            self.img_qkv,
            self.img_qknorm_q,
            self.img_qknorm_k,
            self.img_out,
            self.img_mlp,
            cos_img,
            sin_img,
            cos_kv,
            sin_kv,
        )
        txt_out = self._branch(
            txt_branch_in,
            kv,
            c,
            self.txt_norm,
            self.txt_mod,
            self.txt_qkv,
            self.txt_qknorm_q,
            self.txt_qknorm_k,
            self.txt_out,
            self.txt_mlp,
            None,
            None,
            cos_kv,
            sin_kv,
        )
        return img_out, txt_out
