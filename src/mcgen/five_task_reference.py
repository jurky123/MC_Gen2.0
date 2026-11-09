"""Expand every sampled MC/reference pair into all five supervised tasks."""
from collections import defaultdict
from dataclasses import dataclass
import math
import json

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
    sources: list | None = None

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
        recipe=json.loads((root/'recipe.json').read_text()) if (root/'recipe.json').exists() else {}
        self.five_sampler = FiveTaskSampler(self.rows,recipe.get('source_hq_fraction',.6))
        self.replay = None
        if (root/'text_replay.json').exists():
            from .text_replay import TextReplayData
            self.replay = TextReplayData(root, self.cfg, self.encoder.identity,encoder=self.encoder)
            paired_hashes = {p['target_hash'] for p in self.rows}
            if any(r['target_hash'] in paired_hashes
                   for ds in self.replay.datasets.values() for r in ds.records):
                raise ValueError('Replay overlaps paired targets')

    def sample_plan(self, recipe, rng):
        plan = self.five_sampler.sample(recipe['pairs_per_update'], recipe['unconditional_per_update'], rng)
        ntext, nnull = recipe.get('replay_text_per_update', 0), recipe.get('replay_null_per_update', 0)
        if ntext or nnull:
            if self.replay is None:
                raise ValueError('Replay requested without a prepared text catalog')
            from .text_replay import append_replay
            plan = append_replay(plan, self.replay, ntext, nnull, rng)
        if len(plan) != recipe['batch_size']:
            raise ValueError('Task counts do not match the logical batch')
        return plan

    def plan_batch(self, plan, device):
        if plan.sources is None:
            return self.batch(plan.ids, plan.tasks, plan.variants, device)
        from .text_replay import merge_task_batches
        paired = [j for j, s in enumerate(plan.sources) if s == 'paired']
        replay = [j for j, s in enumerate(plan.sources) if s == 'replay']
        a = self.batch([plan.ids[j] for j in paired], [plan.tasks[j] for j in paired],
            [plan.variants[j] for j in paired], device) if paired else None
        b = self.replay.batch([plan.ids[j] for j in replay], [plan.tasks[j] for j in replay], device) if replay else None
        return merge_task_batches(plan, a, b)


def configure_trainable(model, freeze_time_mlp=False):
    model.requires_grad_(True).eval()
    if freeze_time_mlp:
        model.time.requires_grad_(False)


def learning_rates(update, recipe):
    warm = min(1., (update+1)/recipe['warmup_steps'])
    late = max(0., (update-recipe['decay_start'])/max(1, recipe['steps']-recipe['decay_start']))
    factor = recipe['lr_floor']+(1-recipe['lr_floor'])*(1+math.cos(math.pi*min(late, 1.)))/2
    rates=(recipe['base_lr']*warm*factor, recipe['visual_lr']*warm*factor)
    return rates+(recipe['text_lr']*warm*factor,) if 'text_lr' in recipe else rates


def noisy_batch(target, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    t = torch.rand(len(target), generator=rng, device=device)
    eps = torch.randn(target.shape, generator=rng, device=device)
    return target*(1-t[:, None, None, None])+eps*t[:, None, None, None], t, eps-target


def accumulate_loss(model, batch, noisy, microbatch, *, autocast=True,sort_by_length=False):
    """Exactly weight every view once, including the final short microbatch."""
    if microbatch < 1:
        raise ValueError('Positive microbatch required')
    _, text, mask, refs, present = batch
    x, t, velocity = noisy
    n = len(x)
    order=torch.argsort(mask.sum(-1),stable=True) if sort_by_length else torch.arange(n,device=x.device)
    values=x.new_empty(n)
    start=0
    while start<n:
        stop = min(start+microbatch, n)
        if sort_by_length:
            counts=mask.sum(-1)
            first=int(counts[order[start]])
            bound=0 if first==0 else 2**max(0,(first-1).bit_length())
            while stop>start+1 and int(counts[order[stop-1]])>bound:
                stop-=1
        ids=order[start:stop]
        r = None if refs is None else refs[ids]
        p = None if present is None else present[ids]
        m=mask[ids]
        length=max(1,int(m.nonzero()[:,-1].max())+1) if m.any() else 1
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
            ctx = model.context(text[ids,:length], m[:,:length], r, p)
            prediction = model(x[ids], t[ids], ctx)
            per_sample = (prediction.float()-velocity[ids]).square().mean((1, 2, 3))
        (per_sample.sum()/n).backward()
        values[ids]=per_sample.detach()
        start=stop
    return values
