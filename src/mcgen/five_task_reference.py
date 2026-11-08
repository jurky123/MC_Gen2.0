"""Expand every sampled MC/reference pair into all five supervised tasks."""
from collections import defaultdict
from dataclasses import dataclass
import math

import torch

from .reference_recovery import RecoveryData

TASK_VIEWS = (
    ('joint_short', 'joint', 0),
    ('joint_detail', 'joint', 1),
    ('text_short', 'text', 0),
    ('text_detail', 'text', 1),
    ('reference', 'reference', 0),
)


@dataclass
class TaskPlan:
    ids: list
    tasks: list
    variants: list
    names: list
    pair_slots: list

    def __len__(self):
        return len(self.ids)


class FiveTaskSampler:
    def __init__(self, rows, hq_fraction=.6):
        if not 0 <= hq_fraction <= 1:
            raise ValueError('Invalid source fraction')
        self.hq_fraction = hq_fraction
        self.groups = {}
        for quality in ('highest_quality', 'reviewed'):
            targets = defaultdict(list)
            for i, row in enumerate(rows):
                if row['split'] == 'train' and row['quality_tier'] == quality:
                    targets[row['target_hash']].append(i)
            self.groups[quality] = list(targets.values())
        if not self.groups['highest_quality']:
            raise ValueError('HQ training targets required')

    def sample(self, pairs, unconditional, rng):
        if pairs < 1 or unconditional < 0:
            raise ValueError('Invalid task expansion size')

        def pick():
            quality = 'highest_quality'
            if self.groups['reviewed'] and torch.rand((), generator=rng) >= self.hq_fraction:
                quality = 'reviewed'
            pool = self.groups[quality]
            target = pool[int(torch.randint(len(pool), (), generator=rng))]
            return target[int(torch.randint(len(target), (), generator=rng))]

        entries = []
        for slot in range(pairs):
            i = pick()
            entries.extend((i, task, variant, name, slot) for name, task, variant in TASK_VIEWS)
        entries.extend((pick(), 'unconditional', 0, 'unconditional', -1) for _ in range(unconditional))
        permutation = torch.randperm(len(entries), generator=rng).tolist()
        values = list(zip(*(entries[j] for j in permutation)))
        return TaskPlan(*(list(x) for x in values))


class FiveTaskData(RecoveryData):
    def __init__(self, root):
        super().__init__(root)
        self.five_sampler = FiveTaskSampler(self.rows)

    def plan_batch(self, plan, device):
        return self.batch(plan.ids, plan.tasks, plan.variants, device)


def configure_trainable(model, freeze_time_mlp=False):
    model.requires_grad_(True).eval()
    if freeze_time_mlp:
        model.time.requires_grad_(False)


def learning_rates(update, recipe):
    warm = min(1., (update+1)/recipe['warmup_steps'])
    late = max(0., (update-recipe['decay_start'])/max(1, recipe['steps']-recipe['decay_start']))
    factor = recipe['lr_floor']+(1-recipe['lr_floor'])*(1+math.cos(math.pi*min(late, 1.)))/2
    return recipe['base_lr']*warm*factor, recipe['visual_lr']*warm*factor


def noisy_batch(target, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    t = torch.rand(len(target), generator=rng, device=device)
    eps = torch.randn(target.shape, generator=rng, device=device)
    return target*(1-t[:, None, None, None])+eps*t[:, None, None, None], t, eps-target


def accumulate_loss(model, batch, noisy, microbatch, *, autocast=True):
    """Exactly weight every view once, including the final short microbatch."""
    if microbatch < 1:
        raise ValueError('Positive microbatch required')
    _, text, mask, refs, present = batch
    x, t, velocity = noisy
    n = len(x)
    values = []
    for start in range(0, n, microbatch):
        stop = min(start+microbatch, n)
        r = None if refs is None else refs[start:stop]
        p = None if present is None else present[start:stop]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
            ctx = model.context(text[start:stop], mask[start:stop], r, p)
            prediction = model(x[start:stop], t[start:stop], ctx)
            per_sample = (prediction.float()-velocity[start:stop]).square().mean((1, 2, 3))
        (per_sample.sum()/n).backward()
        values.append(per_sample.detach())
    return torch.cat(values)
