"""Caption policy and task sampling for reviewed pairs plus the HQ bundle."""
from collections import defaultdict
import math
import re

import torch


def mixed_training_budget(targets, batch_size, draws_per_target=120):
    """Keep exposure fixed when resolution changes the feasible batch size."""
    if targets <= 0 or batch_size <= 0 or draws_per_target <= 0:
        raise ValueError('Positive target, batch and exposure counts required')
    return math.ceil(targets*draws_per_target/batch_size)


def mixed_validation_ids(rows, limit, seed=20261107):
    """Fill the validation budget even when a source family was excluded."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row['quality_tier'] == 'highest_quality' and row['split'] == 'val':
            groups[row['source_family']].append(i)
    for family, pool in groups.items():
        order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)).tolist()
        groups[family] = [pool[i] for i in order]
    result = []
    while groups and len(result) < limit:
        for family in sorted(list(groups)):
            if groups[family]:
                result.append(groups[family].pop(0))
                if len(result) == limit:
                    break
            else:
                del groups[family]
    if not result:
        raise ValueError('No HQ validation samples')
    return result


def target_captions(row):
    """Use MC observations/verified semantics, keeping HD captions separate."""
    name = ' '.join(row['name_en'].split())
    description = ' '.join(row.get('mc_annotation_en', '').split())
    description = re.sub(r'^Minecraft pixel item icon:\s*', '', description, flags=re.I)
    description = re.split(r'\b(?:Preserve the described|Keep the complete|Without a wearer)\b',
                           description, maxsplit=1, flags=re.I)[0].strip()
    description = description.rstrip(' .')
    detailed = description if name.casefold() in description.casefold() else f'{name}. {description}'
    return list(dict.fromkeys(x for x in (name, detailed.strip(' .')) if x))


class MixedSampler:
    """Reference targets balanced by source; pure text only from HQ MC targets."""
    tasks = ('joint', 'reference', 'text', 'unconditional')

    def __init__(self, rows, mix=None, split='train'):
        self.rows = rows
        self.mix = mix or {'joint': .70, 'reference': .10, 'text': .18, 'unconditional': .02}
        if set(self.mix) != set(self.tasks) or any(v < 0 for v in self.mix.values()) or abs(sum(self.mix.values())-1) > 1e-8:
            raise ValueError('Explicit joint/reference/text/unconditional mix required')
        self.groups = {}
        for family in ('reviewed', 'highest_quality'):
            targets = defaultdict(list)
            for i, row in enumerate(rows):
                if row['split'] == split and row['quality_tier'] == family:
                    targets[row['target_hash']].append(i)
            self.groups[family] = list(targets.values())
        if not self.groups['highest_quality']:
            raise ValueError('HQ training/validation targets required')
        self.weights = torch.tensor([self.mix[t] for t in self.tasks])

    def sample(self, n, rng):
        modes = torch.multinomial(self.weights, n, replacement=True, generator=rng).tolist()
        ids, tasks, variants = [], [], []
        for mode in modes:
            task = self.tasks[mode]
            family = 'highest_quality'
            if task in ('joint', 'reference') and self.groups['reviewed'] and torch.rand((), generator=rng) >= .6:
                family = 'reviewed'
            pool = self.groups[family]
            group = pool[int(torch.randint(len(pool), (), generator=rng))]
            i = group[int(torch.randint(len(group), (), generator=rng))]
            labels = self.rows[i]['captions']
            # Prefer the observed/verified MC description, retain the short name.
            variant = len(labels)-1 if len(labels) > 1 and torch.rand((), generator=rng) < .7 else 0
            ids.append(i)
            tasks.append(task)
            variants.append(variant)
        return ids, tasks, variants


def initialize_mixed(config, state, device):
    """Copy all learned weights; resize only the fixed spatial position buffer."""
    from .config import Experiment
    from .model import PixelFlow
    import torch.nn.functional as F
    source = Experiment.from_dict(state['config'])
    before, after = source.to_dict()['model'], config.to_dict()['model']
    for key in ('size', 'ref_size'):
        before.pop(key)
        after.pop(key)
    if before != after or not config.model.visual_adapter or config.train.trainable != 'joint_finetune':
        raise ValueError('Joint transfer requires the same spatial adapter architecture')
    weights = dict(state['model'])
    if source.model.size != config.model.size:
        grid = weights['position'].transpose(1, 2).reshape(1, config.model.width, source.model.size, source.model.size)
        weights['position'] = F.interpolate(grid, size=(config.model.size, config.model.size),
                                           mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
    model = PixelFlow(config.model).to(device)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(True).eval()
    return model


class MixedData:
    def __init__(self, root):
        import hashlib
        import json
        import numpy as np
        from PIL import Image
        from .catalog import file_hash
        from .config import Experiment
        from .disk_text import DiskTextEncoder
        from .images import content_hash, encode, native_grid
        from .tasks import conditioned_prompts
        self.root = root
        self.cfg = Experiment.load(root/'config.yaml')
        self.rows = json.loads((root/'pairs.json').read_text())
        self.pairs_sha256 = file_hash(root/'pairs.json')
        audit = json.loads((root/'data_audit.json').read_text())
        if audit['pairs_sha256'] != self.pairs_sha256 or audit['review_snapshot_sha256'] != file_hash(root/'review_snapshot.json'):
            raise ValueError('Training data/selection changed')
        meta = json.loads((root/'visual_features/metadata.json').read_text())
        if not meta['complete'] or meta['pairs_sha256'] != self.pairs_sha256:
            raise ValueError('Incomplete or mismatched visual cache')
        self.visual_identity = meta['visual_identity']
        self.features = np.memmap(root/'visual_features/features.uint16.mmap', mode='r',
                                  dtype='uint16', shape=tuple(meta['shape']))
        if len(self.rows) != len(self.features) or set(meta['chunks']) != {str(i) for i in range(0, len(self.rows), 32)}:
            raise ValueError('Missing visual cache chunks')
        for offset, digest in meta['chunks'].items():
            start = int(offset)
            if hashlib.sha256(np.asarray(self.features[start:start+32]).tobytes()).hexdigest() != digest:
                raise ValueError('Visual features changed')
        self.encoder = DiskTextEncoder(root/'text_features')
        text_meta = json.loads((root/'text_features/metadata.json').read_text())
        for name, key in [('features.uint16.mmap', 'features_sha256'), ('masks.uint8.mmap', 'masks_sha256')]:
            if file_hash(root/'text_features'/name) != text_meta[key]:
                raise ValueError('Text features changed')
        arrays, texts, masks = [], [], []
        for row in self.rows:
            with Image.open(row['target_file']) as im:
                pixels = native_grid(im, self.cfg.model.size)
            if content_hash(pixels) != row['target_hash']:
                raise ValueError('MC target changed')
            arrays.append(encode(pixels))
            labels = row['captions']
            h, m = self.encoder.encode(conditioned_prompts(labels, self.cfg), 'cpu')
            texts.append(torch.stack((h[0], h[-1])))
            masks.append(torch.stack((m[0], m[-1])))
        self.targets = torch.stack(arrays)
        self.text = torch.stack(texts)
        self.mask = torch.stack(masks)
        owners = {}
        for row in self.rows:
            for kind, key in [('target', row['target_hash']), ('reference', row['reference_sha256'])]:
                old = owners.setdefault((kind, key), row['split'])
                if old != row['split']:
                    raise ValueError('Train/holdout image leakage')
        if any(r['split'] == 'train' and r['quality_tier'] == 'evaluation_only' for r in self.rows):
            raise ValueError('Unreviewed legacy image in training')
        self.sampler = MixedSampler(self.rows, self.cfg.train.task_mix)

    def batch(self, ids, tasks, variants, device):
        import numpy as np
        text = self.text[ids, variants].to(device)
        mask = self.mask[ids, variants].to(device)
        mask &= torch.tensor([task in ('joint', 'text') for task in tasks], device=device)[:, None]
        present = torch.tensor([task in ('joint', 'reference') for task in tasks], device=device)
        refs = torch.from_numpy(np.array(self.features[ids], copy=True)).view(torch.bfloat16).to(device)
        return self.targets[ids].to(device), text, mask, refs, present
