import torch


def _velocity_at(model, x, t, text, cfg, text_uncond, text_mask=None, text_uncond_mask=None):
    v_c = model(x, t, text, text_mask=text_mask)
    if cfg > 0.0 and text_uncond is not None:
        v_u = model(x, t, text_uncond, text_mask=text_uncond_mask)
        v_c = v_u + cfg * (v_c - v_u)
    return v_c


def euler(model, z, text, steps=20, cfg=0.0, text_uncond=None,
          text_mask=None, text_uncond_mask=None):
    # Data lives at t=0 and noise at t=1 (x_t = (1-t)x0 + t z), so generation
    # integrates the reverse ODE from t=1 down to t=0: x <- x - dt * v.
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=z.dtype)
        v = _velocity_at(model, x, t, text, cfg, text_uncond, text_mask, text_uncond_mask)
        x = x - dt * v
    return x


def heun(model, z, text, steps=20, cfg=0.0, text_uncond=None,
         text_mask=None, text_uncond_mask=None):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=z.dtype)
        t_next = torch.full((x.shape[0],), max(0.0, 1.0 - (i + 1) * dt),
                            device=x.device, dtype=z.dtype)
        v = _velocity_at(model, x, t, text, cfg, text_uncond, text_mask, text_uncond_mask)
        x_e = x - dt * v
        v2 = _velocity_at(model, x_e, t_next, text, cfg, text_uncond, text_mask, text_uncond_mask)
        x = x - 0.5 * dt * (v + v2)
    return x


def midpoint(model, z, text, steps=20, cfg=0.0, text_uncond=None,
             text_mask=None, text_uncond_mask=None):
    return heun(model, z, text, steps, cfg, text_uncond, text_mask, text_uncond_mask)


SOLVERS = {"euler": euler, "heun": heun, "midpoint": midpoint}


def sample(model, z, text, steps=20, cfg=0.0, text_uncond=None, solver="heun",
           text_mask=None, text_uncond_mask=None):
    x = SOLVERS[solver](model, z, text, steps=steps, cfg=cfg, text_uncond=text_uncond,
                        text_mask=text_mask, text_uncond_mask=text_uncond_mask)
    return x.clamp(-1.0, 1.0)


def to_uint8(x):
    return ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
