from collections import Counter
import copy

import torch

from mcgen.five_task_reference import (FiveTaskSampler, TASK_VIEWS, accumulate_loss,
                                       configure_trainable, learning_rates, noisy_batch)
from mcgen.model import PixelFlow
from mcgen.smoke import tiny_config


def rows():
    return [{'split': split, 'quality_tier': quality, 'target_hash': str(i)}
            for i, (split, quality) in enumerate([
                ('train', 'highest_quality'), ('train', 'reviewed'),
                ('val', 'highest_quality'), ('test', 'highest_quality'),
                ('val', 'evaluation_only')])]


def test_every_pair_gets_both_joint_both_text_and_image_without_holdout_leakage():
    sampler = FiveTaskSampler(rows())
    plan = sampler.sample(31, 5, torch.Generator().manual_seed(771))
    assert len(plan) == 160 and set(plan.ids) == {0, 1}
    assert Counter(plan.names) == {**{n: 31 for n, _, _ in TASK_VIEWS}, 'unconditional': 5}
    for slot in range(31):
        positions = [j for j, x in enumerate(plan.pair_slots) if x == slot]
        assert len({plan.ids[j] for j in positions}) == 1
        assert {(plan.names[j], plan.tasks[j], plan.variants[j]) for j in positions} == set(TASK_VIEWS)
    assert all(v in (0, 1) for v in plan.variants)
    assert any(plan.ids[j] == 1 for j, task in enumerate(plan.tasks) if task == 'text')
    again = sampler.sample(31, 5, torch.Generator().manual_seed(771))
    assert plan == again


def test_source_balance_applies_to_every_task_including_text():
    sampler = FiveTaskSampler(rows())
    plan = sampler.sample(4096, 0, torch.Generator().manual_seed(557))
    for name, _, _ in TASK_VIEWS:
        ids = [i for i, n in zip(plan.ids, plan.names) if n == name]
        assert .56 < ids.count(0)/len(ids) < .64


def test_full_update_budget_does_not_freeze_text_or_decay_too_early():
    r = {'steps': 16000, 'warmup_steps': 300, 'decay_start': 12000,
         'base_lr': 2e-6, 'visual_lr': 2e-5, 'lr_floor': .3}
    assert learning_rates(0, r)[0] > 0
    assert learning_rates(10000, r) == (2e-6, 2e-5)
    assert learning_rates(15999, r)[0] >= .3*2e-6


def test_hq_oversampling_and_text_learning_rate_are_explicit():
    sampler=FiveTaskSampler(rows(),hq_fraction=.8)
    plan=sampler.sample(4096,0,torch.Generator().manual_seed(19))
    assert .77<plan.ids.count(0)/len(plan)<.83
    recipe={'steps':12000,'warmup_steps':500,'decay_start':8000,'base_lr':1e-6,'visual_lr':1e-5,'text_lr':1e-5,'lr_floor':.2}
    assert learning_rates(1000,recipe)==(1e-6,1e-5,1e-5)
    assert len(learning_rates(0,recipe))==3


def test_length_grouped_microbatches_keep_losses_and_gradients_aligned():
    torch.manual_seed(38)
    cfg=copy.deepcopy(tiny_config().model)
    cfg.conditioning='text_only';cfg.byte_tokens=False;cfg.text_dim=12;cfg.text_length=16
    model=PixelFlow(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():p.add_(torch.randn_like(p)*.03)
    other=copy.deepcopy(model)
    target=torch.randn(7,4,cfg.size,cfg.size)
    mask=torch.arange(16)[None]<torch.tensor([0,15,2,8,3,1,10])[:,None]
    batch=(target,torch.randn(7,16,12),mask,None,None)
    noisy=noisy_batch(target,177,'cpu')
    a=accumulate_loss(model,batch,noisy,7,autocast=False)
    b=accumulate_loss(other,batch,noisy,3,autocast=False,sort_by_length=True)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6,msg=name)


def test_microbatch_accumulation_matches_full_batch_with_short_tail():
    torch.manual_seed(19)
    cfg = copy.deepcopy(tiny_config().model)
    cfg.conditioning = 'text_only'
    cfg.visual_adapter = True
    cfg.visual_dim = 12; cfg.visual_grid = 2; cfg.visual_layers = [0, 1]
    model = PixelFlow(cfg).eval()
    # Exercise nonzero attention/visual gates and a nonzero output projection.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p)*.03)
    other = copy.deepcopy(model)
    n = 7
    target = torch.randn(n, 4, cfg.size, cfg.size)
    batch = (target, torch.randn(n, 1, cfg.text_dim),
             torch.tensor([[True], [False], [True], [True], [False], [True], [False]]),
             torch.randn(n, 4, 12), torch.tensor([True, True, False, True, False, False, True]))
    noisy = noisy_batch(target, 117, 'cpu')
    assert not torch.equal(noisy[0][0], noisy[0][1])
    full = accumulate_loss(model, batch, noisy, n, autocast=False)
    split = accumulate_loss(other, batch, noisy, 3, autocast=False)
    torch.testing.assert_close(full, split, rtol=1e-5, atol=1e-6)
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_text_and_visual_learning_preserve_frozen_time_mlp_through_optimizer_step():
    torch.manual_seed(118)
    cfg = copy.deepcopy(tiny_config().model)
    cfg.conditioning = 'text_only'; cfg.visual_adapter = True
    cfg.visual_dim = 12; cfg.visual_grid = 2; cfg.visual_layers = [0, 1]
    model = PixelFlow(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p)*.03)
    configure_trainable(model, True)
    original_time = {k: x.detach().clone() for k, x in model.time.state_dict().items()}
    original_text = model.text.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    target = torch.randn(4, 4, cfg.size, cfg.size)
    batch = (target, torch.randn(4, 1, cfg.text_dim), torch.ones(4, 1, dtype=torch.bool),
             torch.randn(4, 4, 12), torch.tensor([True, False, True, False]))
    accumulate_loss(model, batch, noisy_batch(target, 211, 'cpu'), 3, autocast=False)
    assert all(p.grad is None for p in model.time.parameters())
    assert model.text.weight.grad.norm().item() > 0
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in model.visual_adapter.parameters())
    opt.step()
    assert all(torch.equal(original_time[k], x) for k, x in model.time.state_dict().items())
    assert not torch.equal(original_text, model.text.weight)
