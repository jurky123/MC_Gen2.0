import pytest
import torch
import torch.nn.functional as F

from mcgen.overfit_diagnostics import (CATEGORIES, recover_target_pixels,
    same_category_permutation, select_training_targets, target_pixel_tokens)


def test_oracle_preserves_pixels_even_after_adapter_layernorm():
    target = torch.rand(3,4,16,16)*2-1
    tokens = target_pixel_tokens(target)
    assert tokens.shape == (3,64,4096)
    assert torch.equal(recover_target_pixels(tokens), target)
    assert torch.allclose(recover_target_pixels(F.layer_norm(tokens,(4096,)), normalized=True),
                          target, atol=1e-6, rtol=1e-6)


def test_uniform_brightness_is_not_erased_by_oracle_normalization():
    target = torch.stack([torch.full((4,16,16),v) for v in (-1.,0.,1.)])
    tokens = F.layer_norm(target_pixel_tokens(target),(4096,))
    restored = recover_target_pixels(tokens, normalized=True)
    assert torch.allclose(target, restored, atol=1e-6)
    assert not torch.equal(tokens[0],tokens[1])


def test_negative_reference_controls_stay_within_category_without_identity():
    categories = [category for category in CATEGORIES for _ in range(4)]
    perm = same_category_permutation(categories)
    assert sorted(perm.tolist()) == list(range(16))
    assert all(i != j and categories[i] == categories[j] for i,j in enumerate(perm.tolist()))
    with pytest.raises(ValueError):
        same_category_permutation(['helmet'])


def test_selection_uses_only_unique_hq_train_images():
    rows = [{'split':split,'quality_tier':quality,'target_hash':f'{category}-{i}',
             'captions':[f'item {category}']} for category in CATEGORIES
            for i,(split,quality) in enumerate([('train','highest_quality')]*5+
                [('val','highest_quality'),('test','highest_quality'),('train','reviewed')])]
    rows.append(dict(rows[0]))
    selected = select_training_targets(rows)
    assert len(selected) == len({rows[i]['target_hash'] for i,_ in selected}) == 16
    assert all(rows[i]['split']=='train' and rows[i]['quality_tier']=='highest_quality' for i,_ in selected)
    assert selected == select_training_targets(rows)
