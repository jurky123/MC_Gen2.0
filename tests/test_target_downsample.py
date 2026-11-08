"""Dataset resizing must preserve exact grids and handle transparent RGB safely."""
import numpy as np
import pytest

from mcgen.images import rgba
from mcgen.target_downsample import METHODS, to_native16


@pytest.mark.parametrize('method', METHODS)
def test_exact_double_pixels_roundtrip_without_changing_content(method):
    rng = np.random.default_rng(11)
    original = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    original = rgba(original)
    source = original.repeat(2, 0).repeat(2, 1)
    result, lossless = to_native16(source, method)
    assert lossless
    assert np.array_equal(result, original)
    assert np.array_equal(result.repeat(2, 0).repeat(2, 1), source)


def test_area_average_does_not_mix_in_hidden_background_color():
    source = np.zeros((32, 32, 4), dtype=np.uint8)
    source[:2, :2] = [0, 255, 0, 0]
    source[0, 0] = [220, 20, 10, 255]
    output, lossless = to_native16(source, 'box')
    assert not lossless
    assert output[0, 0].tolist() == [220, 20, 10, 64]
    assert not output[1:, :, 3].any()


def test_palette_candidate_keeps_a_thin_part_and_uses_only_source_colors():
    source = np.zeros((32, 32, 4), dtype=np.uint8)
    source[0, 0] = [13, 74, 190, 255]
    source[4:7, 4:7] = [218, 130, 30, 200]
    output, lossless = to_native16(source, 'palette')
    assert not lossless
    assert output[0, 0].tolist() == [13, 74, 190, 255]
    source_pixels = {tuple(x) for x in source.reshape(-1, 4)}
    assert {tuple(x) for x in output.reshape(-1, 4)} <= source_pixels


def test_lossy_conversion_is_explicit_and_source_is_not_mutated():
    source = np.zeros((32, 32, 4), dtype=np.uint8)
    source[7:22, 9:26] = [80, 160, 230, 255]
    before = source.copy()
    for method in METHODS:
        output, lossless = to_native16(source, method)
        assert output.shape == (16, 16, 4) and output.dtype == np.uint8
        assert not lossless
    assert np.array_equal(source, before)
    with pytest.raises(ValueError, match='Unknown'):
        to_native16(source, 'typo')
    with pytest.raises(ValueError, match='32x32'):
        to_native16(np.zeros((31, 32, 4), dtype=np.uint8))
