"""Native-pixel flow Transformer. Text/reference projections are shared across depth.

One token per native pixel: a 16px target has 256 tokens, and a 32px export is
only nearest-neighbour presentation. Explicit null tokens handle all conditions.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def position(size, width):
    freq = torch.exp(-math.log(10000) * torch.arange(width // 4) / max(width // 4 - 1, 1))
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x, y = x.flatten()[:, None] * freq, y.flatten()[:, None] * freq
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1)[None]


class Block(nn.Module):
    def __init__(self, width, heads, conditioning="shared", reference_encoding="spatial"):
        super().__init__()
        self.conditioning = conditioning
        self.reference_encoding = reference_encoding
        if reference_encoding == "dual":
            self.attribute_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            # A new adapter initially leaves a pretrained backbone unchanged.
            self.attribute_gate = nn.Parameter(torch.zeros(width))
        if conditioning == "separate":
            self.reference_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.reference_gate = nn.Parameter(torch.zeros(width))
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, time, context, mask):
        values = self.modulation(time).chunk(9, -1)
        for i in range(3):
            shift, scale, gate = (v[:, None] for v in values[i * 3:i * 3 + 3])
            h = self.norms[i](x) * (1 + scale) + shift
            if i == 0:
                h = self.self_attention(h, h, h, need_weights=False)[0]
            elif i == 1:
                if self.conditioning == "separate":
                    text, ref = context[:2]
                    text_mask, ref_mask = mask[:2]
                    ht = self.cross_attention(h, text, text, key_padding_mask=~text_mask, need_weights=False)[0]
                    hr = self.reference_attention(h, ref, ref, key_padding_mask=~ref_mask, need_weights=False)[0]
                    h = ht + self.reference_gate.sigmoid() * hr
                    if self.reference_encoding == "dual":
                        attributes, attribute_mask = context[2], mask[2]
                        ha = self.attribute_attention(h, attributes, attributes,
                            key_padding_mask=~attribute_mask, need_weights=False)[0]
                        present = attribute_mask[:, 1:].any(-1)[:, None, None]
                        h = h + present * self.attribute_gate.tanh() * ha
                else:
                    h = self.cross_attention(h, context, context, key_padding_mask=~mask, need_weights=False)[0]
            else:
                h = self.ff(h)
            x = x + gate.tanh() * h
        return x


class PixelFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.input = nn.Linear(4, w)
        self.text = nn.Linear(cfg.text_dim, w)
        if cfg.multimodal_adapter:
            self.image_adapter = nn.Sequential(nn.LayerNorm(cfg.text_dim), nn.Linear(cfg.text_dim, 1024),
                                               nn.GELU(), nn.Linear(1024, w))
            nn.init.zeros_(self.image_adapter[-1].weight)
            nn.init.zeros_(self.image_adapter[-1].bias)
        if cfg.byte_tokens:
            self.byte_embedding = nn.Embedding(258, cfg.text_dim, padding_idx=0)
            self.byte_position = nn.Parameter(torch.randn(1, cfg.text_length, cfg.text_dim) * 0.01)
        self.text_null = nn.Parameter(torch.randn(1, 1, w) * 0.02)
        self.ref_null = nn.Parameter(torch.randn(1, 1, w) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(2, 1, w) * 0.02)
        ref_layers = [nn.Conv2d(4, w // 2, 3, padding=1), nn.SiLU()]
        ratio, channels = cfg.ref_size // cfg.size, w // 2
        while ratio > 1:
            stride = 2 if ratio % 2 == 0 else ratio
            ref_layers.extend([nn.Conv2d(channels, w, 3, stride=stride, padding=1), nn.SiLU()])
            ratio //= stride
            channels = w
        ref_layers.append(nn.Conv2d(channels, w, 1))
        if cfg.conditioning != "text_only":
            self.reference_encoder = nn.Sequential(*ref_layers)
        if cfg.reference_encoding == "dual":
            # Pointwise features + symmetric reductions: no coordinates, patches
            # or neighbourhoods can transmit the reference silhouette.
            self.attribute_encoder = nn.Sequential(nn.Conv2d(4, w // 4, 1), nn.SiLU(),
                                                   nn.Conv2d(w // 4, w, 1))
        self.register_buffer("position", position(cfg.size, w))
        self.time = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([Block(w, cfg.heads, cfg.conditioning, cfg.reference_encoding) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(w)
        self.output = nn.Linear(w, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if cfg.visual_adapter:
            from .reference_adapter import SpatialReferenceAdapter
            self.visual_adapter = SpatialReferenceAdapter(cfg)

    def context(self, text, text_mask, reference, ref_present, reference_mode=None):
        b = text.shape[0]
        mode = reference_mode if reference_mode is not None else ("attributes" if self.cfg.reference_encoding == "dual" else "shape")
        modes = [mode] * b if isinstance(mode, str) else list(mode)
        if len(modes) != b or any(m not in ("attributes", "shape") for m in modes):
            raise ValueError("one attributes/shape reference mode required per sample")
        if self.cfg.reference_encoding != "dual" and any(m != "shape" for m in modes):
            raise ValueError("attributes mode requires a dual reference checkpoint")
        if text.dtype == torch.long:
            h = self.byte_embedding(text) + self.byte_position[:, :text.shape[1]]
        else:
            h = text
        h = h.float()
        # Raw Qwen intermediate states can contain a shared BOS outlier more
        # than 100x larger than content tokens. Normalize each token before
        # projection; keep this opt-in so historical checkpoints retain their
        # original computation. No trainable parameters or cached projections.
        if self.cfg.text_normalization == "layernorm":
            h = F.layer_norm(h, (h.shape[-1],))
        adapter_input = h
        h = self.text(h) + self.type_embedding[0]
        if self.cfg.multimodal_adapter and ref_present is not None:
            active = ref_present.bool() & text_mask.any(-1)
            if active.any():
                h = h + self.image_adapter(adapter_input) * active[:, None, None]
        # Masked tokens are erased, so null conditions cannot leak their values.
        h = h.masked_fill(~text_mask[..., None], 0)
        null_t = self.text_null.expand(b, -1, -1)
        valid_t = torch.ones(b, 1, dtype=torch.bool, device=h.device)
        if self.cfg.conditioning == "text_only":
            if self.cfg.visual_adapter and reference is not None:
                if reference.shape != (b, self.cfg.visual_grid**2, self.cfg.visual_dim):
                    raise ValueError('reference tokens differ from visual adapter contract')
                present = (torch.ones(b, dtype=torch.bool, device=h.device) if ref_present is None
                           else ref_present.to(device=h.device, dtype=torch.bool))
                if present.shape != (b,):
                    raise ValueError('one reference presence flag required per sample')
                if reference_mode is not None:
                    raise ValueError('visual tokens do not use attributes/shape modes')
                if bool(present.any()):
                    ref = self.visual_adapter.project(reference)
                    return torch.cat((null_t, h), 1), torch.cat((valid_t, text_mask), 1), (ref, present)
            if reference is not None and (ref_present is None or bool(ref_present.any())):
                raise ValueError("text_only model does not accept references")
            return torch.cat((null_t, h), 1), torch.cat((valid_t, text_mask), 1)
        # Reference preprocessing is done once by runtime/training, tokens once
        # per sampling call. No zero-image convention or stochastic model dropout.
        if reference is None:
            reference = torch.zeros(b, 4, self.cfg.ref_size, self.cfg.ref_size, device=h.device)
            ref_present = torch.zeros(b, dtype=torch.bool, device=h.device)
        if ref_present is None:
            ref_present = torch.ones(b, dtype=torch.bool, device=h.device)
        if reference.shape[1:] != (4, self.cfg.ref_size, self.cfg.ref_size):
            raise ValueError("reference input size differs from model contract")
        ref = self.reference_encoder(reference.float()).flatten(2).transpose(1, 2) + self.position + self.type_embedding[1]
        shape_present = ref_present & torch.tensor([m == "shape" for m in modes], device=h.device)
        ref = ref.masked_fill(~shape_present[:, None, None], 0)
        mask_r = shape_present[:, None].expand(-1, ref.shape[1])
        if self.cfg.conditioning == "separate":
            contexts = (torch.cat((null_t, h), 1), torch.cat((self.ref_null.expand(b, -1, -1), ref), 1))
            masks = (torch.cat((valid_t, text_mask), 1), torch.cat((valid_t, mask_r), 1))
            if self.cfg.reference_encoding == "dual":
                features = self.attribute_encoder(reference.float()).flatten(2).float()
                attributes = torch.stack((features.mean(-1), features.var(-1, unbiased=False), features.amax(-1)), 1)
                attributes = attributes.masked_fill(~ref_present[:, None, None], 0)
                contexts += (torch.cat((torch.zeros_like(null_t), attributes), 1),)
                masks += (torch.cat((valid_t, ref_present[:, None].expand(-1, 3)), 1),)
            return contexts, masks
        context = torch.cat((null_t, h, self.ref_null.expand(b, -1, -1), ref), 1)
        mask = torch.cat((valid_t, text_mask, valid_t, mask_r), 1)
        return context, mask

    def forward(self, x, t, context):
        if x.shape[1:] != (4, self.cfg.size, self.cfg.size):
            raise ValueError("model input does not match native grid")
        tokens = self.input(x.flatten(2).transpose(1, 2)) + self.position
        freq = torch.exp(-math.log(10000) * torch.arange(self.cfg.width // 2, device=x.device) / (self.cfg.width // 2))
        phase = t[:, None].float() * freq[None] * 1000
        time = self.time(torch.cat((phase.cos(), phase.sin()), -1))
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, time, *context[:2])
            if self.cfg.visual_adapter and len(context) == 3:
                tokens = self.visual_adapter.inject(i, tokens, *context[2])
        return self.output(self.norm(tokens)).transpose(1, 2).reshape_as(x)
