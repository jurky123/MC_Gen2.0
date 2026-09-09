import torch
import torch.nn as nn
import torch.nn.functional as F

from .double_stream import DoubleStreamBlock
from .single_stream import SingleStreamBlock
from .modulation import TimestepEmbedder
from .norm import RMSNorm
from .rope2d import Rotary2D, pad_rope


class MCFlowDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.text_dim = cfg.text_dim
        self.max_text_tokens = cfg.max_text_tokens
        self.patch_dim = cfg.patch_dim

        self.patch_embed = nn.Linear(self.patch_dim, cfg.hidden_size, bias=cfg.bias)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_size, bias=cfg.bias)
        self.text_null = nn.Parameter(torch.zeros(cfg.text_dim))
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    text_mlp_ratio=cfg.text_mlp_ratio,
                    bias=cfg.bias,
                    qk_norm=cfg.qk_norm,
                )
                for _ in range(cfg.double_stream_blocks)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    bias=cfg.bias,
                    qk_norm=cfg.qk_norm,
                )
                for _ in range(cfg.single_stream_blocks)
            ]
        )

        self.head_norm = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, self.patch_dim, bias=cfg.bias)

        self.rope = Rotary2D(self.head_dim)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(B, gh * gw, C * p * p), gh, gw

    def _depatchify(self, tokens, H, W):
        B, N, d = tokens.shape
        p = self.patch_size
        C = self.in_channels
        tokens = tokens.view(B, C * p * p, N)
        return F.fold(tokens, output_size=(H, W), kernel_size=p, stride=p)

    def _drop_text(self, text_emb, cond_drop_prob):
        B = text_emb.shape[0]
        mask = torch.rand(B, 1, device=text_emb.device) < cond_drop_prob
        if mask.any():
            null = self.text_null.view(1, 1, -1).expand_as(text_emb)
            text_emb = torch.where(mask[:, :, None], null, text_emb)
        return text_emb

    def forward(self, x, t, text, cond_drop_prob=None, return_tokens=False):
        B, C, H, W = x.shape
        p = self.patch_size
        tokens, gh, gw = self._patchify(x)
        img = self.patch_embed(tokens)

        Lt = text.shape[1]
        if cond_drop_prob is not None and self.training and cond_drop_prob > 0:
            text = self._drop_text(text, cond_drop_prob)
        text_emb = self.text_proj(text)

        c = self.t_embedder(t)

        cos_img, sin_img = self.rope.get(gh, gw, x.device, x.dtype)
        cos_kv, sin_kv = pad_rope(cos_img, sin_img, Lt)

        for block in self.double_blocks:
            img, txt = block(img, text_emb, c, cos_img, sin_img, cos_kv, sin_kv)

        seq = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            seq = block(seq, c, cos_kv, sin_kv)

        img_tokens = seq[:, Lt:]
        h = self.head_norm(img_tokens)
        vel_tokens = self.head(h)
        vel = self._depatchify(vel_tokens, H, W)
        if return_tokens:
            return vel, img_tokens
        return vel

    def param_count(self):
        return sum(p.numel() for p in self.parameters())