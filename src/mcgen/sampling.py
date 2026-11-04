"""Single ODE/CFG implementation shared by evaluation, CLI and API."""
import math
import torch


@torch.no_grad()
def sample(model, text, mask, reference=None, ref_present=None, *, seeds, steps=20,
           text_scale=2.0, ref_scale=2.0, solver="heun", reference_mode=None, embedding_adapter=False):
    if steps < 1 or solver not in ("euler", "heun"):
        raise ValueError("positive steps and euler/heun solver required")
    if any(not math.isfinite(v) or v < 0 for v in (text_scale, ref_scale)):
        raise ValueError("CFG scales must be finite and nonnegative")
    if len(seeds) != text.shape[0]:
        raise ValueError("one seed required per prompt")
    device = next(model.parameters()).device
    cfg = model.cfg
    x = torch.cat([torch.randn(1, 4, cfg.size, cfg.size,
                              generator=torch.Generator(device=device).manual_seed(int(seed)), device=device)
                   for seed in seeds], 0)
    null_mask = torch.zeros_like(mask)
    nn = model.context(text, null_mask, None, None)
    if embedding_adapter and not cfg.multimodal_adapter:
        raise ValueError("checkpoint has no trained multimodal adapter")
    adapter_present = torch.ones(len(text), dtype=torch.bool, device=device) if embedding_adapter else None
    tn = model.context(text, mask, None, adapter_present) if text_scale else None
    has_ref = reference is not None and (ref_present is None or bool(ref_present.any())) and ref_scale > 0
    # Independent controls: text guidance at no reference; reference guidance
    # adds conditional-reference effect. If text_scale=0, truly image-only.
    if has_ref:
        if text_scale:
            tt = model.context(text, mask, reference, ref_present, reference_mode=reference_mode)
        else:
            tt = model.context(text, null_mask, reference, ref_present, reference_mode=reference_mode)

    def velocity(value, time):
        t = torch.full((value.shape[0],), time, device=device)
        v0 = model(value, t, nn)
        vt = model(value, t, tn) if tn is not None else v0
        out = v0 + text_scale * (vt - v0)
        if has_ref:
            out = out + ref_scale * (model(value, t, tt) - vt)
        return out

    for i in range(steps):
        t, dt = 1 - i / steps, 1 / steps
        v = velocity(x, t)
        proposed = x - dt * v
        x = proposed if solver == "euler" else x - 0.5 * dt * (v + velocity(proposed, max(0., t - dt)))
    return x.clamp(-1, 1)
