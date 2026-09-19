import torch
import torch.nn.functional as F

from .flow import reconstruct_x0


def flow_mse(v_pred, target, reduction="mean"):
    return F.mse_loss(v_pred, target, reduction=reduction)


def seam_edges(x, border_width=1):
    l = x[..., :, :border_width]
    r = x[..., :, -border_width:]
    t = x[..., :border_width, :]
    b = x[..., -border_width:, :]
    return (l - r).abs().mean(), (t - b).abs().mean()


def tile_loss(x0_hat, border_width=2):
    e1, e2 = seam_edges(x0_hat, border_width)
    return 0.5 * (e1 + e2)


def flow_tile_loss(v_pred, xt, t, target, tile_cfg, tileable_mask=None):
    """Flow MSE + optional per-sample masked seam/tile loss.

    ``tileable_mask`` (bool tensor, per sample) restricts the tile loss to
    tileable textures (P1-1): items/tools/armor must not be pushed toward
    edge-matching. When None, the legacy all-sample behaviour is kept.
    """
    loss = flow_mse(v_pred, target)
    if not tile_cfg.get("enabled", True):
        return loss
    weight = float(tile_cfg.get("weight", 0.03))
    max_t = float(tile_cfg.get("max_t", 0.7))
    bw = int(tile_cfg.get("border_width", 2))
    if weight <= 0.0:
        return loss
    keep = t < max_t
    if tileable_mask is not None:
        keep = keep & tileable_mask.to(keep.device)
    if keep.any():
        x0 = reconstruct_x0(xt, t, v_pred)
        tile = tile_loss(x0[keep], border_width=bw)
        loss = loss + weight * tile
    return loss