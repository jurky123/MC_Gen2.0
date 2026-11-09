"""Broad filename-text replay, isolated from the reference task's holdouts."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch

from .catalog import file_hash, read_catalog
from .dataset import CatalogDataset
from .disk_text import load_text_cache
from .tasks import conditioned_prompts

REPLAY_FAMILIES = {
    'food': 'bread meat steak fish salmon cod porkchop beef chicken mutton rabbit potato carrot beetroot berry berries fruit cookie cake pie soup stew sushi cheese candy chocolate cabbage corn onion rice wheat sugar egg mushroom honey toast sandwich bacon burger sausage milk coffee tea juice',
    'plant': 'sapling flower leaf leaves vine grass root roots petal petals seed seeds log wood bark acorn pollen nut crop cactus bamboo',
    'material': 'nugget dust powder shard shards fragment plate rod stick string leather hide feather bone scale claw tooth horn pearl shell fiber thread fabric cloth coal charcoal chunk ore ash ashes clump brick rock rubber sand crystal',
    'component': 'coil wire gear cog circuit chip capacitor cable connector motor engine piston sensor lens spring screw battery cell cartridge wafer module upgrade pump',
    'container': 'bucket bowl bottle flask vial canister can jar bag pouch box barrel tank cup basket sack',
    'utility': 'compass clock map disc record saddle leash name tag torch key mirror binoculars spyglass thermometer whistle ticket blueprint',
    'magic': 'rune scroll sigil totem talisman amulet relic artifact essence soul mana spell orb',
    'additional_tool': 'wrench saw sickle shears knife chisel drill pliers screwdriver brush scoop fishing hammer chainsaw hoe scalpel cutter',
    'additional_weapon': 'gun pistol rifle shotgun bullet grenade bomb rocket rapier saber dagger katana trident shuriken boomerang whip sling slingshot',
    'accessory': 'belt scarf cape cloak shirt coat hat cap bracelet earring badge glove goggles wings wing ring necklace charm crown boot',
}


def stratified_text_subset(records, count=2000, seed=2026110831):
    """Fixed unique targets; broad item types, then diverse unchanged names.

    The old `other` category contains most resources. Split it into broad noun
    families so food, plants and components are covered without treating color
    or material names as hundreds of equally weighted item categories. These
    are sampling strata derived from text, not new target annotations.
    """
    from .reference_recovery import coarse_caption
    train = {}
    for r in records:
        if r['split'] == 'train':
            train.setdefault(r['target_hash'], r)
    if not 1 <= count <= len(train):
        raise ValueError('Insufficient unique original training targets for replay subset')
    noun_family = {noun: family for family, nouns in REPLAY_FAMILIES.items() for noun in nouns.split()}
    types, subtypes = {}, {}
    for key, r in train.items():
        category = r.get('category_15', 'other')
        if category == 'other':
            category = coarse_caption(r['prompt'])
            if category == 'item':
                words = re.findall(r'[a-z]{3,}', r['prompt'].lower())
                noun = next((w for w in reversed(words) if w in noun_family), None)
                category = 'other:'+noun_family[noun] if noun is not None else 'other:misc'
                subtypes[key] = noun or (words[-1] if words else 'misc')
        types[key] = category
    buckets = defaultdict(lambda: defaultdict(list))
    for key, r in train.items():
        category = types[key]
        buckets[category][subtypes.get(key, category)].append(r)
    def order(value):
        return hashlib.sha256(f'{seed}:{value}'.encode()).digest()
    pools = {}
    for category, subtypes_pool in buckets.items():
        lists = []
        for subtype in sorted(subtypes_pool, key=order):
            names = defaultdict(list)
            for r in subtypes_pool[subtype]:
                names[r['prompt']].append(r)
            # One target per distinct original prompt before repeated labels.
            variants = [sorted(names[p], key=lambda r: order(r['target_hash']), reverse=True)
                        for p in sorted(names, key=order)]
            ordered = []
            while variants:
                remaining = []
                for group in variants:
                    ordered.append(group.pop())
                    if group:
                        remaining.append(group)
                variants = remaining
            lists.append(list(reversed(ordered)))
        pool = []
        while lists:
            remaining = []
            for group in lists:
                if group:
                    pool.append(group.pop())
                    if group:
                        remaining.append(group)
            lists = remaining
        pools[category] = iter(pool)
    selected, active = [], sorted(pools)
    while len(selected) < count:
        remaining = []
        for category in active:
            r = next(pools[category], None)
            if r is not None:
                selected.append({**r, 'replay_type': category})
                remaining.append(category)
            if len(selected) == count:
                break
        active = remaining
    result = selected+[r for r in records if r['split'] == 'val']
    report = {'seed': seed, 'train_targets': len(selected), 'eligible_train_targets': len(train),
        'train_unique_prompts': len({r['prompt'] for r in selected}),
        'train_categories': dict(Counter(r.get('category_15', 'other') for r in selected)),
        'train_types': dict(sorted(Counter(r['replay_type'] for r in selected).items())),
        'type_policy': 'original category; other split by broad filename noun family; sampling strata, not visual annotations',
        'selection': 'round-robin types, then subtypes, then distinct original names; immutable seeded order',
        'text_changed': False}
    return result, report


def select_replay_records(records, pairs):
    """Keep original splits/text; remove paired targets and holdout projects.

    An overlap with the original foundation's train split is reported, even
    though excluding it from this continuation cannot undo prior exposure.
    """
    paired_hashes = {r['target_hash'] for r in pairs}
    holdout_hashes = {r['target_hash'] for r in pairs if r['split'] != 'train'}
    held_groups = {r['split_group'] for r in pairs if r['split'] != 'train'}
    held_lineages = {r.get('provenance', {}).get('lineage') for r in pairs if r['split'] != 'train'} - {None}
    for r in records:
        if r['target_hash'] in holdout_hashes:
            held_groups.add(r['split_group'])
            held_lineages.add(r['lineage'])
    kept, excluded, prior = [], Counter(), set()
    # Retain only fields needed by the loader and provenance audit. Images and
    # embeddings stay in the existing immutable mmap files.
    keys = ('id', 'split', 'split_group', 'lineage', 'target', 'target_hash',
            'prompt', 'asset_type', 'tileable', 'alpha_mode', 'category_15',
            'caption_supervision', 'caption_variant', 'source_filename')
    for r in records:
        if r['split'] == 'train' and r['target_hash'] in holdout_hashes:
            prior.add(r['target_hash'])
        if r['split'] not in ('train', 'val') or r['asset_type'] != 'item':
            excluded['outside_original_item_train_val'] += 1
        elif r['target_hash'] in paired_hashes:
            excluded['paired_target_overlap'] += 1
        elif r['split'] == 'train' and (r['split_group'] in held_groups or r['lineage'] in held_lineages):
            excluded['reference_holdout_project'] += 1
        elif not r['prompt'].strip():
            raise ValueError('Empty text in original replay catalog')
        elif r.get('caption_supervision') != 'deterministic_resource_text_only':
            raise ValueError('Replay requires the original filename-only supervision')
        else:
            kept.append({**{k: r[k] for k in keys if k in r}, 'reference': None})
    if not all(any(r['split'] == s for r in kept) for s in ('train', 'val')):
        raise ValueError('Replay needs disjoint original train and validation targets')
    report = {'excluded_records': dict(excluded),
              'reference_holdout_targets_seen_by_foundation': len(prior),
              'prior_exposure_note': 'Exclusion prevents new replay leakage, not earlier foundation exposure.',
              'split_targets': {s: len({r['target_hash'] for r in kept if r['split'] == s}) for s in ('train', 'val')},
              'train_categories': dict(Counter(r.get('category_15', 'other') for r in kept if r['split'] == 'train')),
              'train_unique_prompts': len({r['prompt'] for r in kept if r['split'] == 'train'})}
    return kept, report


class TextReplayData:
    def __init__(self, root, cfg, text_identity,encoder=None):
        self.spec = json.loads((Path(root)/'text_replay.json').read_text())
        recipe = json.loads((Path(root)/'recipe.json').read_text())
        if file_hash(Path(root)/'text_replay.json') != recipe['text_replay_sha256']:
            raise ValueError('Replay specification changed after recipe preparation')
        spec = self.spec
        self.encoder = (encoder if encoder is not None and encoder.root==Path(spec['features']).resolve()
                        else load_text_cache(spec['features'],text_identity))
        if self.encoder.identity!=text_identity:raise ValueError('Replay text encoder identity mismatch')
        for name, digest in spec['feature_hashes'].items():
            if file_hash(Path(spec['features'])/name) != digest:
                raise ValueError('Replay text cache changed: '+name)
        records, meta = read_catalog(spec['catalog'])
        if meta['fingerprint'] != spec['catalog_fingerprint'] or meta['native_size'] != cfg.model.size:
            raise ValueError('Replay catalog contract changed')
        self.cfg = cfg
        self.datasets = {s: CatalogDataset(records, cfg.model.size, cfg.model.ref_size,
            s, asset_type='item', task='text', alpha_mode=cfg.alpha_mode) for s in ('train', 'val')}
        prompts = conditioned_prompts([r['prompt'] for r in records], cfg)
        if any(p not in self.encoder.index for p in prompts):
            raise ValueError('Replay prompt missing from the compatible frozen cache')
        self.groups = self.datasets['train'].groups[False]

    def sample(self, n, rng):
        ids = []
        for g in torch.randint(len(self.groups), (n,), generator=rng).tolist():
            group = self.groups[g]
            ids.append(group[int(torch.randint(len(group), (), generator=rng))])
        return ids

    def validation_indices(self, n, seed=2026110819):
        """Fixed category round-robin; one row per independent original target."""
        groups, seen = defaultdict(list), set()
        for i, r in enumerate(self.datasets['val'].records):
            if r['target_hash'] not in seen:
                groups[r.get('category_15', 'other')].append(i)
                seen.add(r['target_hash'])
        rng = torch.Generator().manual_seed(seed)
        for category, pool in groups.items():
            groups[category] = [pool[j] for j in torch.randperm(len(pool), generator=rng).tolist()]
        out = []
        while len(out) < min(n, len(seen)):
            for category in sorted(groups):
                if groups[category] and len(out) < n:
                    out.append(groups[category].pop())
        return out

    def batch(self, ids, tasks, device, split='train'):
        if len(ids) != len(tasks) or any(t not in ('text', 'unconditional') for t in tasks):
            raise ValueError('Replay supports only text and unconditional tasks')
        b = self.datasets[split].batch(ids, device)
        h, mask = self.encoder.encode(conditioned_prompts(b['prompts'], self.cfg), device)
        mask = mask & torch.tensor([t == 'text' for t in tasks], device=device)[:, None]
        return b['image'], h, mask, None, torch.zeros(len(ids), dtype=torch.bool, device=device)


def append_replay(plan, replay, text_count, null_count, rng):
    from .five_task_reference import TaskPlan
    if min(text_count, null_count) < 0:
        raise ValueError('Replay counts must be nonnegative')
    n = text_count+null_count
    if not n:
        return plan
    ids = replay.sample(n, rng)
    tasks = ['text']*text_count+['unconditional']*null_count
    names = ['broad_text']*text_count+['broad_unconditional']*null_count
    combined = TaskPlan(plan.ids+ids, plan.tasks+tasks, plan.variants+[0]*n,
        plan.names+names, plan.pair_slots+[-1]*n, ['paired']*len(plan)+['replay']*n)
    order = torch.randperm(len(combined), generator=rng).tolist()
    return TaskPlan(*([getattr(combined, key)[j] for j in order]
        for key in ('ids', 'tasks', 'variants', 'names', 'pair_slots', 'sources')))


def merge_task_batches(plan, paired_batch, replay_batch):
    """Restore shuffled task order; replay rows have no reference tokens."""
    blocks = []
    if paired_batch is not None:
        blocks.append(([i for i, s in enumerate(plan.sources) if s == 'paired'], paired_batch))
    if replay_batch is not None:
        blocks.append(([i for i, s in enumerate(plan.sources) if s == 'replay'], replay_batch))
    if sum(len(p) for p, _ in blocks) != len(plan):
        raise ValueError('Missing task source')
    outputs = []
    for k in range(5):
        example = next((b[k] for _, b in blocks if b[k] is not None), None)
        if example is None:
            outputs.append(None)
            continue
        shape=list(example.shape[1:])
        if k in (1,2):shape[0]=max(b[k].shape[1] for _,b in blocks)
        result = example.new_zeros((len(plan), *shape))
        for positions, b in blocks:
            if b[k] is not None:
                if k in (1,2):result[positions,:b[k].shape[1]]=b[k]
                else:result[positions] = b[k]
        outputs.append(result)
    return tuple(outputs)
