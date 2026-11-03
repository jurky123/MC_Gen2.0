import torch
import torch.nn as nn
import torch.nn.functional as F

from .double_stream import DoubleStreamBlock
from .single_stream import SingleStreamBlock
from .cross_attention import CrossAttnBlock
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

        # cross-attention text injection (image queries -> text keys/values)
        self.text_injection = getattr(cfg, "text_injection", "joint")
        self.cross_blocks = nn.ModuleList([
            CrossAttnBlock(cfg.hidden_size, cfg.num_heads, cfg.text_dim,
                           mlp_ratio=cfg.mlp_ratio, bias=cfg.bias, qk_norm=cfg.qk_norm)
            for _ in range(int(getattr(cfg, "cross_attn_blocks", 0)))
        ])

        self.rope = Rotary2D(self.head_dim)

        # ---- optional reference-image conditioning (HD->MC Stylizer) ----
        self.ref_condition = str(getattr(cfg, "ref_condition", "") or "")
        if self.ref_condition:
            from .reference import (
                ReferenceCrossAttention,
                ReferenceEncoder,
                SpatialAdapter,
            )

            self.ref_size = int(getattr(cfg, "ref_size", 64))
            # Learned null-reference token (one vector broadcast over positions):
            # the model ALWAYS runs the reference branch, with either real
            # reference tokens or this null token, so training and CFG-time
            # "no reference" mean exactly the same thing.
            self.ref_null = nn.Parameter(torch.zeros(cfg.hidden_size))
            self.ref_encoder = ReferenceEncoder(
                cfg.hidden_size, in_channels=int(getattr(cfg, "ref_channels", 4)),
                ref_size=self.ref_size, bias=cfg.bias)
            self.ref_adapter_positions = list(getattr(cfg, "ref_adapter_positions", [0, 1]))
            if "adapter" in self.ref_condition:
                self.ref_adapters = nn.ModuleList([
                    SpatialAdapter(cfg.hidden_size, bias=cfg.bias)
                    for _ in self.ref_adapter_positions])
            n_ref_cross = int(getattr(cfg, "ref_cross_blocks", 0))
            # positions = indices into single_stream_blocks after which a
            # reference cross-attention is applied; -1 (or >= len) means
            # "after all blocks" (the v3 layout). Padded/truncated to
            # n_ref_cross so a partial list still covers every block.
            pos = list(getattr(cfg, "ref_cross_positions", [-1]))
            if n_ref_cross:
                pos = (pos + [-1] * n_ref_cross)[:n_ref_cross]
            self.ref_cross_positions = pos
            if "cross" in self.ref_condition and n_ref_cross:
                self.ref_cross_blocks = nn.ModuleList([
                    ReferenceCrossAttention(
                        cfg.hidden_size, cfg.num_heads, bias=cfg.bias,
                        qk_norm=cfg.qk_norm, mlp_ratio=2.0)
                    for _ in range(n_ref_cross)])

    def encode_reference(self, reference):
        """(B, 4, S, S) in [-1,1] -> (B, G*G, D) reference tokens."""
        return self.ref_encoder(reference.to(dtype=self.patch_embed.weight.dtype))

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
        # (B, N, C*p*p) -> (B, C*p*p, N): transpose token/channel axes, then fold.
        tokens = tokens.transpose(1, 2).reshape(B, C * p * p, N)
        return F.fold(tokens, output_size=(H, W), kernel_size=p, stride=p)

    def _drop_text(self, text_emb, cond_drop_prob):
        B = text_emb.shape[0]
        mask = torch.rand(B, 1, device=text_emb.device) < cond_drop_prob
        if mask.any():
            null = self.text_null.view(1, 1, -1).expand_as(text_emb)
            text_emb = torch.where(mask[:, :, None], null, text_emb)
        return text_emb

    def forward(self, x, t, text, text_mask=None, cond_drop_prob=None,
                reference=None, ref_drop_prob=None, ref_present=None,
                return_tokens=False):
        B, C, H, W = x.shape
        p = self.patch_size
        tokens, gh, gw = self._patchify(x)
        img = self.patch_embed(tokens)

        c = self.t_embedder(t)

        # Reference branch is always active: real reference tokens when
        # available, otherwise the learned null-reference token (also used for
        # per-sample dropout and for CFG). This removes the previous
        # inconsistency where a *zero image* was used as the null condition
        # (the encoder's affine norm made encoder(0) != 0).
        ref_tokens = None
        # Legacy checkpoints (v1/v2/v3) trained with "zero image == null" and
        # no learned null token: keep their exact behaviour for evaluation.
        legacy_null = getattr(self, "legacy_null", False)
        if self.ref_condition and not (legacy_null and reference is None):
            present = None
            if reference is not None:
                ref_tokens = self.encode_reference(reference)
                n_ref = ref_tokens.shape[1]
                present = torch.ones(B, dtype=torch.bool, device=x.device)
                if ref_present is not None:
                    present = ref_present.to(device=x.device, dtype=torch.bool)
                if ref_drop_prob is not None and self.training and ref_drop_prob > 0:
                    present = present & (torch.rand(B, device=x.device) >= ref_drop_prob)
            else:
                n_ref = gh * gw
            null = self.ref_null.view(1, 1, -1).expand(B, n_ref, -1).to(img.dtype)
            if ref_tokens is None:
                ref_tokens = null
            elif present is None or present.all():
                pass
            else:
                ref_tokens = torch.where(present[:, None, None], ref_tokens, null)
        adapter_idx = {pos: k for k, pos in enumerate(
            getattr(self, "ref_adapter_positions", []))}

        if self.text_injection == "cross_attn":
            # The MMDiT image/text streams keep a learned register token (in
            # Stage 1 the text stream was always this constant null), and the
            # real text token sequence is injected through cross-attention.
            reg = self.text_proj(self.text_null.view(1, 1, -1)).to(img.dtype)
            reg = reg.expand(B, 1, -1)
            lt = 1
            cos_img, sin_img = self.rope.get(gh, gw, x.device, x.dtype)
            cos_cross, sin_cross = pad_rope(cos_img, sin_img, lt)
            if ref_tokens is not None and 0 in adapter_idx:
                img = self.ref_adapters[adapter_idx[0]](img, ref_tokens)
            for block in self.double_blocks:
                img, reg = block(img, reg, c, cos_img, sin_img, cos_cross, sin_cross)
            if ref_tokens is not None and 1 in adapter_idx:
                img = self.ref_adapters[adapter_idx[1]](img, ref_tokens)
            seq = torch.cat([reg, img], dim=1)
            # Each ref-cross block runs exactly once: either right after its
            # mapped single-stream block, or once at the end if its position
            # is -1 / out of range. (Looking the tail map up by block index
            # instead of position used to run early blocks a second time.)
            has_ref_cross = ref_tokens is not None and hasattr(self, "ref_cross_blocks")
            n_sb = len(self.single_blocks)
            mid_cross, tail_cross = {}, []
            if has_ref_cross:
                for k, p in enumerate(self.ref_cross_positions):
                    p = int(p)
                    if 0 <= p < n_sb:
                        mid_cross.setdefault(p, []).append(k)
                    else:
                        tail_cross.append(k)
            for bi, block in enumerate(self.single_blocks):
                seq = block(seq, c, cos_cross, sin_cross)
                if bi in mid_cross:
                    img_part = seq[:, lt:]
                    for k in mid_cross[bi]:
                        img_part = self.ref_cross_blocks[k](img_part, ref_tokens)
                    seq = torch.cat([seq[:, :lt], img_part], dim=1)
            img_tokens = seq[:, lt:]

            key_mask = None
            if text is not None:
                # Match the parameter dtype so the model also works outside an
                # autocast context (e.g. validation / eager sampling).
                text = text.to(dtype=self.patch_embed.weight.dtype)
                if text_mask is None:
                    key_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=x.device)
                else:
                    key_mask = text_mask
                if cond_drop_prob is not None and self.training and cond_drop_prob > 0:
                    keep = (torch.rand(B, device=x.device) >= cond_drop_prob)
                    key_mask = key_mask & keep[:, None]
                for block in self.cross_blocks:
                    img_tokens = block(img_tokens, text, key_mask, c)

            # blocks mapped to position -1 (or beyond the last index) run here
            if has_ref_cross and tail_cross:
                for k in tail_cross:
                    img_tokens = self.ref_cross_blocks[k](img_tokens, ref_tokens)
        else:
            Lt = text.shape[1]
            if cond_drop_prob is not None and self.training and cond_drop_prob > 0:
                text = self._drop_text(text, cond_drop_prob)
            text_emb = self.text_proj(text)

            cos_img, sin_img = self.rope.get(gh, gw, x.device, x.dtype)
            cos_kv, sin_kv = pad_rope(cos_img, sin_img, Lt)

            for block in self.double_blocks:
                img, text_emb = block(img, text_emb, c, cos_img, sin_img, cos_kv, sin_kv)

            seq = torch.cat([text_emb, img], dim=1)
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