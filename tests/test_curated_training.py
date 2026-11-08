"""HQ captions, split-safe task mixing, and real text-only parameter updates."""
import copy

import pytest
import torch

from mcgen.config import Experiment
from mcgen.curated_training import (MixedSampler, initialize_mixed, target_captions,
                                   mixed_training_budget, mixed_validation_ids)
from mcgen.model import PixelFlow
from mcgen.smoke import tiny_config


def test_mc_caption_never_substitutes_hd_or_generation_intent():
    row = {'name_en': 'Sun Gem', 'mc_annotation_en': 'A gold gem with a red center. Preserve the described arrangement. Keep the complete visible silhouette.',
           'hd_annotation_en': 'An elaborate blue dragon', 'expanded_generation_prompt': 'Generate a photograph of a dragon'}
    labels = target_captions(row)
    assert labels[0] == 'Sun Gem'
    assert 'gold gem' in labels[1] and 'red center' in labels[1]
    assert 'Preserve' not in labels[1] and 'dragon' not in ' '.join(labels)


def test_text_tasks_only_use_hq_and_holdouts_never_enter_training():
    rows = [{'quality_tier': family, 'split': split, 'target_hash': str(i), 'captions': ['name', 'detail']}
            for i, (family, split) in enumerate([('reviewed', 'train'), ('reviewed', 'train'),
                ('highest_quality', 'train'), ('highest_quality', 'train'), ('highest_quality', 'val'),
                ('highest_quality', 'test'), ('evaluation_only', 'val')])]
    sampler = MixedSampler(rows)
    ids, tasks, variants = sampler.sample(4096, torch.Generator().manual_seed(19))
    assert set(ids) == {0, 1, 2, 3}
    assert set(tasks) == set(sampler.tasks)
    assert all(rows[i]['quality_tier'] == 'highest_quality' for i, t in zip(ids, tasks)
               if t in ('text', 'unconditional'))
    assert .15 < sum(t in ('text', 'unconditional') for t in tasks)/len(tasks) < .25
    assert .60 < sum(rows[i]['quality_tier'] == 'highest_quality' for i in ids)/len(ids) < .76
    assert sum(v == 1 for v in variants) > len(variants)*.6


def test_native_resolution_transfer_preserves_weights_and_text_loss_learns():
    torch.set_num_threads(1)
    data = tiny_config().to_dict()
    data['model'].update(conditioning='text_only', byte_tokens=False, text_dim=8, text_length=1,
        visual_adapter=True, visual_dim=12, visual_grid=2, visual_layers=[0, 1])
    data['text'].update(backend='qwen3_vl_embedding', model='fake', revision='a'*40)
    data.update(conditioning_input='visual_tokens', task='joint')
    data['train']['trainable'] = 'visual_adapter'
    source_cfg = Experiment.from_dict(data)
    source = PixelFlow(source_cfg.model)
    with torch.no_grad():
        source.output.weight.normal_(std=.1)
        for b in source.blocks:
            b.modulation[-1].bias.fill_(.3)
    new = copy.deepcopy(data)
    new['model'].update(size=16, ref_size=16)
    new['train']['trainable'] = 'joint_finetune'
    cfg = Experiment.from_dict(new)
    state = {'config': source_cfg.to_dict(), 'model': source.state_dict()}
    model = initialize_mixed(cfg, state, 'cpu')
    assert model.position.shape == (1, 256, 32)
    assert source.position.shape == (1, 64, 32)
    for key, weight in source.state_dict().items():
        if key != 'position':
            assert torch.equal(weight, model.state_dict()[key])
    h, mask = torch.randn(2, 1, 8), torch.ones(2, 1, dtype=torch.bool)
    before = model.text.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    result = model(torch.randn(2, 4, 16, 16), torch.rand(2), model.context(h, mask, None, None))
    result.square().mean().backward()
    assert model.text.weight.grad.abs().sum() > 0
    assert model.visual_adapter.projection[-1].weight.grad is None
    opt.step()
    assert not torch.equal(before, model.text.weight)


def test_sampler_rejects_missing_hq_pool():
    with pytest.raises(ValueError, match='HQ'):
        MixedSampler([{'quality_tier': 'reviewed', 'split': 'train', 'target_hash': 'a'}])


def test_single_remaining_family_fills_validation_without_using_train_or_test():
    rows = [{'quality_tier': 'highest_quality', 'source_family': 'gear', 'split': split}
            for split in ['train']*30+['val']*72+['test']*56]
    selected = mixed_validation_ids(rows, 24)
    assert len(selected) == len(set(selected)) == 24
    assert all(rows[i]['split'] == 'val' for i in selected)
    assert selected == mixed_validation_ids(rows, 24)
    assert len(mixed_validation_ids(rows, 100)) == 72
    with pytest.raises(ValueError, match='validation'):
        mixed_validation_ids(rows[:30], 24)


def test_large_batch_does_not_silently_multiply_training_exposure():
    for batch in (64, 128, 512, 1024):
        steps = mixed_training_budget(864, batch, 120)
        assert 864*120 <= steps*batch < 864*120+batch
    with pytest.raises(ValueError, match='Positive'):
        mixed_training_budget(0, 128)


def test_same_resolution_transfer_keeps_original_position_and_all_weights():
    data = tiny_config().to_dict()
    data['model'].update(conditioning='text_only', byte_tokens=False, text_dim=8,
        text_length=1, visual_adapter=True, visual_dim=12, visual_grid=2, visual_layers=[0,1])
    data['text'].update(backend='qwen3_vl_embedding', model='fake', revision='a'*40)
    data.update(conditioning_input='visual_tokens', task='joint')
    data['train']['trainable'] = 'visual_adapter'
    source_cfg = Experiment.from_dict(data)
    source = PixelFlow(source_cfg.model)
    target = copy.deepcopy(data)
    target['train']['trainable'] = 'joint_finetune'
    model = initialize_mixed(Experiment.from_dict(target),
                             {'config': source_cfg.to_dict(), 'model': source.state_dict()}, 'cpu')
    for name, value in source.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
