"""Explicit native16 dataset derivatives; never relax the lossless grid contract."""
import numpy as np
from PIL import Image

from .images import native_grid, rgba

METHODS = ('nearest', 'box', 'palette')


def to_native16(value, method='nearest'):
    """Return (straight RGBA16, lossless).

    Exact repeated cells always take precedence. Lossy conversion only accepts
    real32 targets and must be recorded as such by the dataset builder.
    """
    if method not in METHODS:
        raise ValueError('Unknown target downsample method')
    source = rgba(value)
    try:
        return native_grid(source, 16), True
    except ValueError:
        if source.shape[:2] != (32, 32):
            raise ValueError('Lossy conversion requires a32x32 target') from None
    if method == 'nearest':
        return rgba(np.array(Image.fromarray(source).resize((16, 16), Image.Resampling.NEAREST))), False
    blocks = source.reshape(16, 2, 16, 2, 4).transpose(0, 2, 1, 3, 4).reshape(16, 16, 4, 4)
    colors = blocks[..., :3].astype(np.float64)
    alpha = blocks[..., 3:].astype(np.float64)/255.
    if method == 'box':
        coverage = alpha.mean(-2)
        premul = (colors*alpha).mean(-2)
        result = np.zeros((16, 16, 4), dtype=np.float64)
        np.divide(premul, coverage, out=result[..., :3], where=coverage > 0)
        result[..., 3:] = coverage*255
        return rgba(np.rint(result).clip(0, 255).astype(np.uint8)), False
    # Pick an actual pixel from each2x2 block. The most opaque candidates keep
    # thin parts; proximity to the alpha-weighted mean represents local color.
    # This can thicken edges; it is a visible candidate, not a silent cleanup.
    total = alpha.sum(-2)
    center = np.zeros((16, 16, 3), dtype=np.float64)
    np.divide((colors*alpha).sum(-2), total, out=center, where=total > 0)
    distances = ((colors-center[..., None, :])**2).sum(-1)
    most_opaque = blocks[..., 3].max(-1, keepdims=True)
    distances[blocks[..., 3] != most_opaque] = np.inf
    chosen = distances.argmin(-1)
    result = np.take_along_axis(blocks, chosen[..., None, None], axis=2)[..., 0, :]
    return rgba(result), False
