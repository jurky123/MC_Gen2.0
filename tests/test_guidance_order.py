from types import SimpleNamespace

import pytest
import torch

from mcgen.sampling import sample


class InteractionVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(size=2, multimodal_adapter=False)

    def context(self, text, mask, reference, present, **kwargs):
        t = mask.any(-1).float()
        r = torch.zeros_like(t) if reference is None else present.float()
        # Nonadditive joint response distinguishes the two guidance paths.
        return .01+.02*t+.03*r+.04*t*r

    def forward(self, x, t, context):
        return torch.ones_like(x)*context[:, None, None, None]


def draw(order, ts, rs, reference=True):
    return sample(InteractionVelocity(), torch.zeros(1, 1, 8), torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 4, 8) if reference else None, torch.tensor([True]) if reference else None,
        seeds=[1], steps=1, solver='euler', text_scale=ts, ref_scale=rs,
        initial_noise=torch.zeros(1, 4, 2, 2), guidance_order=order)


def test_reference_first_guides_text_difference_with_image_present():
    torch.testing.assert_close(draw('reference_first', 2., .5), torch.full((1, 4, 2, 2), -.145))
    torch.testing.assert_close(draw('text_first', 2., .5), torch.full((1, 4, 2, 2), -.085))


@pytest.mark.parametrize('ts,rs', [(1., 1.), (2., 2.), (0., 2.), (2., 0.)])
def test_equal_scales_and_single_condition_reductions_match_legacy(ts, rs):
    torch.testing.assert_close(draw('reference_first', ts, rs), draw('text_first', ts, rs))


def test_no_reference_uses_original_text_path():
    assert torch.equal(draw('reference_first', 2., 2., False), draw('text_first', 2., 2., False))


def test_unknown_order_fails():
    with pytest.raises(ValueError, match='guidance_order'):
        draw('invalid', 2., 2.)
