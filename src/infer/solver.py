import torch


def _velocity_at(model, x, t, text, cfg, text_uncond):
    v_c = model(x, t, text)
    if cfg > 0.0 and text_uncond is not None:
        v_u = model(x, t, text_uncond)
        v_c = v_u + cfg * (v_c - v_u)
    return v_c


def euler(model, z, text, steps=20, cfg=0.0, text_uncond=None):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=z.dtype)
        v = _velocity_at(model, x, t, text, cfg, text_uncond)
        x = x + dt * v
    return x


def heun(model, z, text, steps=20, cfg=0.0, text_uncond=None):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=z.dtype)
        v = _velocity_at(model, x, t, text, cfg, text_uncond)
        t_half = t - 0.5 * dt
        x_e = x + 0.5 * dt * v
        v2 = _velocity_at(model, x_e, t_half, text, cfg, text_uncond)
        x = x + dt * v2
    return x


def midpoint(model, z, text, steps=20, cfg=0.0, text_uncond=None):
    return heun(model, z, text, steps, cfg, text_uncond)


SOLVERS = {"euler": euler, "heun": heun, "midpoint": midpoint}


def sample(model, z, text, steps=20, cfg=0.0, text_uncond=None, solver="heun"):
    x = SOLVERS[solver](model, z, text, steps=steps, cfg=cfg, text_uncond=text_uncond)
    return x.clamp(-1.0, 1.0)


def to_uint8(x):
    return ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)