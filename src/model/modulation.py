import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        emb = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


class Modulation(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        s = self.linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = s.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, bias=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(dim, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
