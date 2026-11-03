"""Explicit counterfactual color data utilities; never implicit augmentation.

Hue changes preserve native geometry and alpha, but do not model material edits.
"""
import math
import re
import numpy as np

from .images import rgba


def category_from_caption(prompt):
    """Conservative weak noun labels, not a substitute for shape annotation.

    Material words such as emerald are deliberately not object categories.
    Reject compound products rather than relabeling apple pie as an apple.
    """
    text = prompt.lower()
    exclusions = {
        'pickaxe': r'\b(?:pickaxe head|pickaxe handle|pickaxe mold|pickaxe cast)\b',
        'sword': r'\b(?:sword blade|sword handle|sword mold|sword cast)\b',
        'axe': r'\b(?:axe head|axe handle|axe mold|axe cast)\b',
        'shovel': r'\b(?:shovel head|shovel handle|shovel mold|shovel cast)\b',
        'key': r'\b(?:gate|lock|keyboard|keyhole|keychain|fragment)\b',
        'potion': r'\b(?:cake|frosting|arrow|food|bucket|powder)\b',
        'apple': r'\b(?:pie|juice|jam|jelly|seed|sapling|cake|slice|sauce)\b',
        'flower': r'\b(?:pot|seed|sapling|planter)\b',
        'gem': r'\b(?:ore|dust|nugget|block|stick|hoe|scythe|sickle|sword|axe|shovel|helmet|chestplate|boots|leggings|bow|arrow|plate|ingot|apple|door|key|staff|wand|shield|torch|circuit|rod|tool)\b',
        'ingot': r'\b(?:cast|mold|nugget|dust|ore|handle)\b',
        'door': r'\b(?:trap door|door handle|door frame)\b',
    }
    for name in ('pickaxe','sword','axe','shovel','helmet','shield','key','potion','apple','flower','gem','book','door','ingot'):
        pattern = r'\b(?:gem|gemstone|crystal)\b' if name == 'gem' else rf'\b{name}\b'
        if re.search(pattern, text):
            if name in exclusions and re.search(exclusions[name], text):
                return 'other'
            if name == 'potion' and not re.search(r'\b(?:bottle|flask)\b', text):
                return 'other'
            return name
    return 'other'


def hsv_components(image):
    rgb = rgba(image)[..., :3].astype(np.float64) / 255
    high, low = rgb.max(-1), rgb.min(-1)
    chroma = high - low
    safe = np.where(chroma > 0, chroma, 1.)
    r, g, b = np.moveaxis(rgb, -1, 0)
    hue = np.where(high == r, ((g - b) / safe) % 6,
                   np.where(high == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    return np.where(chroma > 0, hue, 0), chroma, high


def shift_hue(image, degrees):
    """Rotate HSV hue, preserving max/min RGB, alpha and pixel coordinates."""
    if not math.isfinite(degrees):
        raise ValueError("hue shift must be finite")
    a = rgba(image)
    if degrees % 360 == 0:
        return a
    hue, chroma, high = hsv_components(a)
    sector = ((hue + degrees / 360) % 1) * 6
    x = chroma * (1 - np.abs(sector % 2 - 1))
    zero = np.zeros_like(x)
    candidates = np.stack([np.stack(c, -1) for c in
        ((chroma, x, zero), (x, chroma, zero), (zero, chroma, x),
         (zero, x, chroma), (x, zero, chroma), (chroma, zero, x))])
    idx = np.minimum(sector.astype(int), 5)
    rgb = np.take_along_axis(candidates, idx[None, ..., None], axis=0)[0]
    rgb += (high - chroma)[..., None]
    a[..., :3] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    return rgba(a)


def color_summary(image):
    """Circular hue over visible chroma; neutral/background pixels have no vote."""
    a = rgba(image)
    hue, chroma, high = hsv_components(a)
    alpha = a[..., 3].astype(float) / 255
    visible = alpha >= .5
    colored = visible & (chroma >= .12) & (high >= .15)
    weight = chroma * alpha * colored
    mass = float(weight.sum())
    vector = (weight * np.exp(2j * np.pi * hue)).sum()
    return {"hue_degrees": float(np.angle(vector) * 180 / np.pi % 360) if mass else None,
            "concentration": float(abs(vector) / mass) if mass else 0.,
            "chromatic_pixels": int(colored.sum()),
            "chromatic_coverage": float(colored.sum() / max(int(visible.sum()), 1)),
            "mean_chroma": float((chroma * alpha).sum() / max(alpha.sum(), 1.))}


def hue_distance(a, b):
    if a is None or b is None:
        return None
    return float(abs((a - b + 180) % 360 - 180))


def compatible_color_pair(target, reference):
    t, r = color_summary(target), color_summary(reference)
    error = hue_distance(t["hue_degrees"], r["hue_degrees"])
    keep = (t["chromatic_pixels"] >= 8 and t["chromatic_coverage"] >= .15
            and r["chromatic_pixels"] >= 32 and t["concentration"] >= .6
            and r["concentration"] >= .6 and error is not None and error <= 30)
    return keep, {"target": t, "reference": r, "hue_error": error}
