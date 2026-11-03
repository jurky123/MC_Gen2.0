"""Canonical reference preprocessing for the Stylizer (single source of truth).

Every inference path (paired eval, review, open-loop, external images) must use
this function so the reference resolution and resampling match training.

Training used:  HD 384 --LANCZOS--> ref_size (64/128) RGBA(alpha=255) -> [-1,1]
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_alpha_composited(path_or_pil, size):
    """Load an image, composite transparency on white, resize to `size`."""
    if isinstance(path_or_pil, (str, Path)):
        im = Image.open(path_or_pil).convert("RGBA")
    else:
        im = path_or_pil.convert("RGBA") if path_or_pil.mode != "RGBA" else path_or_pil
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im.convert("RGB"), mask=im.split()[3])
    return bg.resize((size, size), Image.Resampling.LANCZOS)


def preprocess_reference(path_or_pil, ref_size):
    """-> (4, ref_size, ref_size) float tensor in [-1, 1] with alpha = 255."""
    rgb = np.asarray(load_alpha_composited(path_or_pil, ref_size), dtype=np.uint8)
    arr = np.dstack([rgb, np.full((ref_size, ref_size), 255, np.uint8)])
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0


def preprocess_reference_batch(items, ref_size):
    return torch.stack([preprocess_reference(x, ref_size) for x in items])
