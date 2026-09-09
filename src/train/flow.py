import torch


def rand_timesteps(n, mode="uniform", device="cpu", dtype=torch.float32, std=1.0):
    if mode == "uniform":
        return torch.rand(n, device=device, dtype=dtype)
    if mode == "logit_normal":
        z = torch.randn(n, device=device, dtype=dtype) * std
        return torch.sigmoid(z)
    raise ValueError(f"unknown timestep sampling mode: {mode}")


def _t_broadcast(t, like):
    while t.dim() < like.dim():
        t = t.unsqueeze(-1)
    return t


def interpolate(x0, z, t):
    t = _t_broadcast(t, x0)
    return (1.0 - t) * x0 + t * z


def velocity_target(x0, z):
    return z - x0


def reconstruct_x0(xt, t, v_pred):
    t = _t_broadcast(t, xt)
    return xt - t * v_pred


def sample_data_noise(x0, t, generator=None):
    z = torch.randn_like(x0)
    xt = interpolate(x0, z, t)
    target = velocity_target(x0, z)
    return xt, z, target