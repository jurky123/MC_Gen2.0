"""Controls that distinguish reference learning from text and noise shortcuts."""
import random
import re

import torch

CATEGORIES = ('helmet', 'sword', 'axe', 'shield')


def select_training_targets(rows, per_category=4, seed=20261108):
    rng = random.Random(seed)
    selected, seen = [], set()
    for category in CATEGORIES:
        pattern = r'\b(?:axe|battleaxe|greataxe|waraxe|handaxe)\b' if category == 'axe' else r'\b'+category+r'\b'
        candidates = {}
        for i, row in enumerate(rows):
            if (row['split'] == 'train' and row['quality_tier'] == 'highest_quality'
                    and row['target_hash'] not in seen and re.search(pattern, row['captions'][0], re.I)):
                candidates.setdefault(row['target_hash'], i)
        pool = list(candidates.values())
        rng.shuffle(pool)
        if len(pool) < per_category:
            raise ValueError('Insufficient distinct HQ training targets for '+category)
        for i in pool[:per_category]:
            selected.append((i, category))
            seen.add(rows[i]['target_hash'])
    return selected


def same_category_permutation(categories):
    result = list(range(len(categories)))
    for category in dict.fromkeys(categories):
        ids = [i for i, value in enumerate(categories) if value == category]
        if len(ids) < 2:
            raise ValueError('Need multiple designs per category')
        for i, j in zip(ids, ids[1:]+ids[:1]):
            result[i] = j
    return torch.tensor(result, dtype=torch.long)


def target_pixel_tokens(target):
    """Invertibly pack16px premultiplied RGBA into64 tokens of4096 channels.

    This deliberately leaks the target and is an upper-bound diagnostic only.
    Fixed +/-1 anchors keep information recoverable after token LayerNorm.
    """
    if target.ndim != 4 or target.shape[1:] != (4,16,16):
        raise ValueError('Expected Bx4x16x16 targets')
    b = len(target)
    patches = target.float().reshape(b,4,8,2,8,2).permute(0,2,4,1,3,5).reshape(b,64,16)
    anchors = patches.new_tensor([1.,-1.]).expand(b,64,2)
    return torch.cat((patches,anchors),-1).repeat(1,1,228)[...,:4096].contiguous()


def recover_target_pixels(tokens, normalized=False):
    values = tokens[...,:16]
    if normalized:
        scale = 2/(tokens[...,16:17]-tokens[...,17:18])
        mean = 1-scale*tokens[...,16:17]
        values = values*scale+mean
    b = len(tokens)
    return values.reshape(b,8,8,4,2,2).permute(0,3,1,4,2,5).reshape(b,4,16,16)
