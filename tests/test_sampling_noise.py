"""Explicit noise supports coupled-resolution diagnostics without mutating input."""
from types import SimpleNamespace

import pytest
import torch

from mcgen.sampling import sample


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(size=8)

    def context(self, *args, **kwargs):
        return None

    def forward(self, x, t, context):
        return torch.zeros_like(x)


def test_explicit_noise_overrides_seed_and_is_not_mutated():
    model = ZeroVelocity()
    h, mask = torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool)
    noise = torch.randn(2, 4, 8, 8)
    original = noise.clone()
    result = sample(model, h, mask, seeds=[3, 4], initial_noise=noise, steps=2)
    other = sample(model, h, mask, seeds=[300, 400], initial_noise=noise, steps=2)
    assert torch.equal(result, noise.clamp(-1, 1))
    assert torch.equal(result, other)
    assert torch.equal(noise, original)


def test_default_seed_path_remains_identical_with_explicit_same_noise():
    model = ZeroVelocity()
    h, mask = torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool)
    seeds = [11, 12]
    noise = torch.cat([torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(s)) for s in seeds])
    assert torch.equal(sample(model, h, mask, seeds=seeds, steps=2),
                       sample(model, h, mask, seeds=seeds, steps=2, initial_noise=noise))


@pytest.mark.parametrize('noise', [torch.zeros(2, 4, 16, 16), torch.full((2, 4, 8, 8), float('nan')),
                                  torch.zeros(2, 4, 8, 8, dtype=torch.int64)])
def test_invalid_explicit_noise_is_rejected(noise):
    with pytest.raises(ValueError, match='initial_noise'):
        sample(ZeroVelocity(), torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool),
               seeds=[1, 2], initial_noise=noise)
