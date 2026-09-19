"""RGBA representation helpers.

Training representation can be "straight" (RGBA as stored on disk) or
"premultiplied": ``rgb_pm = rgb * a / 255`` with alpha kept as-is.
Premultiplied gives transparent pixels a deterministic RGB of 0 instead of
whatever the source had behind the mask, which otherwise wastes capacity and
pollutes distance metrics (docs/MC-Gen2_HD-to-MC_Design_v1.0.md section 3.4).

Conventions:
- tensors are (B, C, H, W) float in [-1, 1] (``x/127.5 - 1`` of source 0..255);
- PNG load/save always uses straight RGBA (``unpremultiply`` before saving);
- loss/eval operate in the configured training representation.
"""
import numpy as np
import torch

A_EPS = 0.5  # source-space alpha threshold for "visible"


def premultiply_rgba_torch(x):
    """(B, 4, H, W) straight RGBA in [-1, 1] -> premultiplied in [-1, 1]."""
    if x.shape[1] != 4:
        raise ValueError(f"expected 4 channels, got {x.shape[1]}")
    rgb_src = (x[:, :3] + 1.0) * 127.5
    a_src = (x[:, 3:4] + 1.0) * 127.5
    rgb_pm = rgb_src * (a_src / 255.0)
    return torch.cat([rgb_pm / 127.5 - 1.0, x[:, 3:4]], dim=1)


def unpremultiply_rgba_torch(x):
    """(B, 4, H, W) premultiplied in [-1, 1] -> straight RGBA in [-1, 1]."""
    if x.shape[1] != 4:
        raise ValueError(f"expected 4 channels, got {x.shape[1]}")
    rgb_pm_src = (x[:, :3] + 1.0) * 127.5
    a_src = (x[:, 3:4] + 1.0) * 127.5
    scale = (a_src / 255.0).clamp_min(1.0 / 255.0)
    rgb_src = (rgb_pm_src / scale).clamp(0.0, 255.0) * (a_src > A_EPS).float()
    return torch.cat([rgb_src / 127.5 - 1.0, x[:, 3:4]], dim=1)


def premultiply_rgba_np(arr):
    """(..., 4) uint8 straight RGBA -> premultiplied uint8."""
    arr = np.asarray(arr)
    rgb = arr[..., :3].astype(np.uint16)
    a = arr[..., 3:4].astype(np.uint16)
    return np.concatenate([((rgb * a + 127) // 255).astype(np.uint8), arr[..., 3:4]], axis=-1)


def unpremultiply_rgba_np(arr):
    """(..., 4) uint8 premultiplied -> straight uint8 (alpha=0 -> rgb=0)."""
    arr = np.asarray(arr)
    rgb_pm = arr[..., :3].astype(np.float32)
    a = arr[..., 3:4].astype(np.float32)
    rgb = np.where(a > A_EPS, np.clip(rgb_pm * 255.0 / np.maximum(a, 1.0), 0.0, 255.0), 0.0)
    return np.concatenate([rgb.astype(np.uint8), arr[..., 3:4]], axis=-1)


def rgba_roundtrip_diff(arr):
    """Max abs diff of premultiply -> unpremultiply vs original (uint8)."""
    pm = premultiply_rgba_np(arr)
    back = unpremultiply_rgba_np(pm)
    return np.abs(back.astype(np.int16) - np.asarray(arr).astype(np.int16))
