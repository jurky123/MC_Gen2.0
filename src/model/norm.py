import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x * rms
        if self.elementwise_affine:
            out = out * self.weight
        return out

    def extra_repr(self):
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class QKNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
