from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mcgen.images import encode, decode
from mcgen.sampling import sample


@pytest.mark.parametrize('solver', ['euler', 'heun'])
@pytest.mark.parametrize('steps', [1, 7, 32])
def test_oracle_flow_recovers_rgba_without_color_or_alpha_damage(solver, steps):
    """An exact constant velocity field must reconstruct a known RGBA target."""
    image=np.zeros((16,16,4),dtype=np.uint8)
    image[3:11,5:10]=[37,153,209,255]
    image[7:10,3:5]=[210,60,15,128]
    target=encode(image)[None]
    seed=1234
    noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(seed))
    class Oracle(nn.Module):
        def __init__(self):
            super().__init__();self.anchor=nn.Parameter(torch.zeros(()));self.cfg=SimpleNamespace(size=16)
        def context(self,*args,**kwargs):return None
        def forward(self,x,t,context):return noise-target
    result=sample(Oracle(),torch.zeros(1,4,3),torch.ones(1,4,dtype=torch.bool),
                  seeds=[seed],steps=steps,text_scale=2,solver=solver)
    assert np.array_equal(decode(result[0]),image)
