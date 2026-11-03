from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcgen.color_supervision import shift_hue, color_summary, hue_distance, compatible_color_pair, category_from_caption
from mcgen.images import native_grid


def test_hue_rotation_preserves_geometry_alpha_and_neutrals():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    a[0, :, :3] = 170
    a[a[..., 3] == 0, :3] = 0
    for angle in (0, 30, 60, 120, 240, 330):
        b = shift_hue(a, angle)
        assert np.array_equal(a[..., 3], b[..., 3])
        assert np.array_equal(a[..., :3].max(-1), b[..., :3].max(-1))
        assert np.array_equal(a[..., :3].min(-1), b[..., :3].min(-1))
        assert np.array_equal(a[0], b[0])
        back = shift_hue(b, -angle)
        assert np.abs(back.astype(int) - a.astype(int)).max() <= 1
        enlarged = shift_hue(a.repeat(4, 0).repeat(4, 1), angle)
        assert np.array_equal(native_grid(enlarged, 16), b)
    with pytest.raises(ValueError, match="finite"):
        shift_hue(a, float('nan'))


def test_paired_shift_has_correct_hue_and_neutral_background_cannot_dominate():
    target = np.zeros((16, 16, 4), np.uint8)
    target[4:12, 4:12] = [220, 40, 40, 255]
    ref = np.full((128, 128, 4), 255, np.uint8)
    ref[30:80, 30:80] = [200, 20, 20, 255]
    for degrees in (0, 60, 120, 240, 330):
        t, r = shift_hue(target, degrees), shift_hue(ref, degrees)
        keep, details = compatible_color_pair(t, r)
        assert keep and details['hue_error'] < 1
        assert hue_distance(color_summary(t)['hue_degrees'], degrees) < 1
    assert not compatible_color_pair(target, shift_hue(ref, 120))[0]
    neutral = np.full_like(target, 255)
    assert color_summary(neutral)['hue_degrees'] is None
    assert not compatible_color_pair(neutral, ref)[0]


@pytest.mark.parametrize('prompt', ['gold apple food pie item','magenta glazed flower pot item',
    'emerald stick item','emerald hoe item','potion cake frosting item','magenta key gate item',
    'copper pickaxe head','iron ingot mold'])
def test_compound_objects_are_not_mislabeled(prompt):
    assert category_from_caption(prompt) == 'other'


@pytest.mark.parametrize('prompt,category', [('lapis apple item with a stem','apple'),
    ('emerald pickaxe item','pickaxe'),('pink crystal key item','key'),
    ('silver gem item with round facets','gem'),('potion item with blue liquid in a bottle','potion')])
def test_explicit_object_nouns_win_over_material_words(prompt, category):
    assert category_from_caption(prompt) == category
