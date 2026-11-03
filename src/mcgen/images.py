"""Boundary contract: disk is straight uint8 RGBA; model is premultiplied [-1,1]."""
import hashlib
import numpy as np
from PIL import Image, ImageOps
import torch


def rgba(value):
    if isinstance(value, Image.Image):
        value = np.array(value.convert("RGBA"))
    a = np.array(value, copy=True)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError("expected H×W×4 uint8 straight RGBA")
    a[a[..., 3] == 0, :3] = 0
    return a


def native_grid(value, size):
    """Only collapse exact repeated cells. Never silently resample training targets."""
    a = rgba(value)
    h, w = a.shape[:2]
    if h != w or h < size or h % size:
        raise ValueError(f"target {w}x{h} cannot losslessly map to {size}x{size}")
    scale = h // size
    small = a[::scale, ::scale].copy()
    if not np.array_equal(a, small.repeat(scale, 0).repeat(scale, 1)):
        raise ValueError(f"target does not have an exact {size}x{size} native grid")
    return small


def content_hash(value):
    a = rgba(value)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def encode(value):
    a = torch.from_numpy(rgba(value)).permute(2, 0, 1).float() / 255
    return torch.cat((a[:3] * a[3:4], a[3:4]), 0) * 2 - 1


def decode(value):
    if value.ndim != 3 or value.shape[0] != 4:
        raise ValueError("expected 4×H×W model tensor")
    a = ((value.detach().float().cpu() + 1) / 2).clamp(0, 1)
    rgb = torch.where(a[3:4] > 0.5 / 255, a[:3] / a[3:4].clamp_min(1 / 255), 0)
    out = torch.cat((rgb.clamp(0, 1), a[3:4]), 0)
    return rgba((out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def composite(value, background=255):
    a = rgba(value).astype(np.float32)
    alpha = a[..., 3:4] / 255
    return a[..., :3] * alpha + background * (1 - alpha)


def reference(value, size):
    """Aspect-preserving RGBA fit; transparent padding retains foreground alpha.

    Pillow resizes RGBA through premultiplied channels, avoiding dark fringes.
    RGB images arrive with opaque alpha; no guessed foreground segmentation.
    """
    im = Image.fromarray(rgba(value))
    im = ImageOps.pad(im, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0, 0))
    return encode(np.array(im))


def apply_alpha(value, mode):
    a = rgba(value)
    if mode == "opaque":
        a[..., 3] = 255
    elif mode == "binary":
        a[..., 3] = (a[..., 3] >= 128).astype(np.uint8) * 255
    elif mode != "continuous":
        raise ValueError("unknown alpha mode")
    return rgba(a)


def export(value, scale=1, colors=0):
    """Optional display/palette conversion, separate from raw model evaluation."""
    if not isinstance(scale, int) or scale < 1 or colors not in (0, *range(2, 257)):
        raise ValueError("invalid export scale or palette size")
    a = rgba(value)
    if colors:
        visible = a[..., 3] > 0
        if visible.any():
            strip = Image.fromarray(a[..., :3][visible][None])
            quantized = np.array(strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB"))[0]
            a[..., :3][visible] = quantized
    return Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.Resampling.NEAREST)
