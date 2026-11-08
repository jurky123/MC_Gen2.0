"""Expanded validation and a two-by-two weight/grid inference experiment.

No training, target resampling, encoder changes, or checkpoint mutation. The
post16 diagnostic is not a model trained at16 on the new data.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.curated_training import MixedData
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from train_visual_reference import thumbnail

RUN = Path('outputs/reference_mixed_hq_20261107')
SEEDS = [202611071, 202611072]
VARIANTS = ('pre16', 'pre32', 'post32', 'post16')


def category(prompt):
    words = re.findall(r'[a-z]+', prompt.lower())
    for word in reversed(words):
        if word in ('helmet', 'crown', 'chestplate', 'armor', 'boots', 'sword', 'shield', 'axe', 'pickaxe',
                    'gem', 'crystal', 'bow', 'staff', 'wand', 'bottle', 'arrow', 'ingot'):
            return word
    return 'other'


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'native').mkdir()
    (out/'inputs').mkdir()
    data = MixedData(RUN)
    previous = json.loads((RUN/'generation_checks/002468/panel.json').read_text())
    excluded = {r['target_hash'] for r in previous}
    excluded.update(r['target_hash'] for r in json.loads((RUN/'generation_checks/002468/fixed_text_different_references.json').read_text()))
    rng = random.Random(2026110711)
    selected = []
    quotas = [('train', 'highest_quality', 'mcgen2_items_500', 8),
              ('train', 'highest_quality', 'original_gear_blur500', 8),
              ('train', 'reviewed', 'reviewed_legacy', 8),
              ('val', 'highest_quality', 'mcgen2_items_500', 12),
              ('val', 'highest_quality', 'original_gear_blur500', 12),
              ('val', 'evaluation_only', 'reviewed_legacy', 8)]
    for split, quality, family, count in quotas:
        buckets = defaultdict(list)
        unique = {}
        for i, row in enumerate(data.rows):
            if (row['split'], row['quality_tier'], row['source_family']) == (split, quality, family) and row['target_hash'] not in excluded:
                unique.setdefault(row['target_hash'], []).append(i)
        for variants in unique.values():
            i = rng.choice(variants)
            buckets[category(data.rows[i]['captions'][0])].append(i)
        for pool in buckets.values():
            rng.shuffle(pool)
        chosen = []
        while len(chosen) < count:
            for key in sorted(buckets):
                if buckets[key] and len(chosen) < count:
                    chosen.append(buckets[key].pop())
            if not any(buckets.values()) and len(chosen) < count:
                raise ValueError('Insufficient distinct targets for quota')
        for i in chosen:
            row = data.rows[i]
            excluded.add(row['target_hash'])
            ident = f'{split}_{sum(r["split"]==split for r in selected):02d}'
            selected.append({**row, 'pair_index': i, 'case_id': ident,
                             'category': category(row['captions'][0])})
            with Image.open(row['reference_file']) as im:
                thumbnail(im, 256).save(out/'inputs'/f'{ident}_reference.png')
            with Image.open(row['target_file']) as im:
                im.convert('RGBA').save(out/'inputs'/f'{ident}_target.png')
    atomic_json(out/'samples.json', selected)
    common = ['diamond sword', 'iron pickaxe', 'gold ingot', 'red gem', 'diamond helmet', 'iron axe', 'emerald', 'shield']
    old_text = DiskTextEncoder('outputs/qwen_vl_embedding_large_20261103/features', expected_identity=data.encoder.identity)
    available = [p for p in common if conditioned_prompts([p], data.cfg)[0] in old_text.index]
    h, _ = old_text.encode(conditioned_prompts(available, data.cfg), 'cpu')
    ids = [r['pair_index'] for r in selected]
    features = {'short': data.text[ids, 0], 'detail': data.text[ids, 1],
        'reference': torch.from_numpy(np.array(data.features[ids], copy=True)).view(torch.bfloat16),
        'common_text': h, 'common_prompts': available, 'visual_identity': data.visual_identity,
        'text_identity': data.encoder.identity}
    torch.save(features, out/'features.pt')
    atomic_json(out/'protocol.json', {'counts': dict(Counter(r['split'] for r in selected)),
        'quality_counts': dict(Counter(r['quality_tier'] for r in selected)),
        'selection_seed': 2026110711, 'generation_seeds': SEEDS,
        'selection': 'fresh filename-stratified targets, excludes all previous monitoring panels and fixed-reference examples; no output filtering',
        'modes': ['text', 'reference', 'joint'], 'steps': 32, 'solver': 'heun', 'text_scale': 2., 'reference_scale': 2.,
        'noise': 'draw iid N(0,1) at32; noise16 = 2*average_pool2(noise32), also unit-variance iid Gaussian. Matched spatial randomness across resolutions.',
        'variants': {'pre16': 'reviewed source checkpoint at its trained native16 grid',
                     'pre32': 'exact saved initial32 checkpoint, before any mixed fine-tuning',
                     'post32': 'final mixed checkpoint at native32',
                     'post16': 'final mixed learned weights at16, restoring the source16 position buffer; inference diagnostic only, not a16-trained counterpart'},
        'extra_caption_check': 'post32 uses the detailed MC caption as well as short names for HQ samples',
        'test_used': False, 'holdout_caveat': 'validation excluded from this mixed fine-tuning run; some original MC targets may have been seen during base pretraining',
        'source_pairs_sha256': data.pairs_sha256, 'features_sha256': file_hash(out/'features.pt')})
    print('PREPARED', len(selected), 'targets; common prompts', available, flush=True)


def learned_hash(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        if name == 'position':
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_variant(name, device, out):
    info = json.loads((RUN/'initialization.json').read_text())
    path = Path(info['checkpoint']) if name == 'pre16' else (RUN/'training/initial_native32.pt' if name == 'pre32' else RUN/'training/latest.pt')
    state = load_checkpoint(path)
    cfg = Experiment.from_dict(state['config'])
    weights = dict(state['model'])
    if name == 'post16':
        original = load_checkpoint(info['checkpoint'])
        cfg.model.size = cfg.model.ref_size = 16
        weights['position'] = original['model']['position'].clone()
        del original
    model = PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(weights, strict=True)
    meta = {'checkpoint': str(path), 'step': state['step'], 'native_grid': cfg.model.size,
        'learned_weights_sha256': learned_hash(weights), 'position_sha256': hashlib.sha256(weights['position'].contiguous().numpy().tobytes()).hexdigest(),
        'text_identity': state['text_identity'], 'visual_identity': state['visual_identity']}
    atomic_json(out/f'model_{name}.json', meta)
    del state, weights
    gc.collect()
    return model


def noise_for(seeds, size, device):
    full = torch.cat([torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(s)) for s in seeds])
    return (full if size == 32 else F.avg_pool2d(full, 2)*2).to(device)


@torch.inference_mode()
def generate(out, group, device, variant=None):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.18, device)
    rows = json.loads((out/'samples.json').read_text())
    features = torch.load(out/'features.pt', weights_only=True)
    names = (variant,) if variant else (('pre16', 'post16') if group == 'native16' else ('pre32', 'post32'))
    progress_name = variant or group
    started = time.monotonic()
    for name in names:
        model = load_variant(name, device, out)
        atomic_json(out/f'progress_{progress_name}.json', {'phase': name, 'done': 0, 'seconds': time.monotonic()-started})
        done = 0
        for seed_index, seed in enumerate(SEEDS):
            for mode in ('text', 'reference', 'joint'):
                for start in range(0, len(rows), 12):
                    ids = list(range(start, min(start+12, len(rows))))
                    missing = [i for i in ids if not (out/'native'/f'{rows[i]["case_id"]}_{name}_{mode}_s{seed_index}.png').exists()]
                    if not missing:
                        continue
                    h = features['short'][missing].to(device)
                    mask = torch.ones(h.shape[:2], dtype=torch.bool, device=device)
                    if mode == 'reference':
                        mask[:] = False
                    reference = None if mode == 'text' else features['reference'][missing].to(device)
                    actual = sample(model, h, mask, reference, seeds=[seed]*len(missing), steps=32,
                                    initial_noise=noise_for([seed]*len(missing), model.cfg.size, device))
                    for j, i in enumerate(missing):
                        Image.fromarray(decode(actual[j])).save(out/'native'/f'{rows[i]["case_id"]}_{name}_{mode}_s{seed_index}.png')
                    done += len(missing)
                    atomic_json(out/f'progress_{progress_name}.json', {'phase': name, 'mode': mode, 'seed_index': seed_index,
                        'done': done, 'seconds': time.monotonic()-started})
            common = features['common_text'].to(device)
            result = sample(model, common, torch.ones(common.shape[:2], dtype=torch.bool, device=device),
                seeds=[seed]*len(common), steps=32, initial_noise=noise_for([seed]*len(common), model.cfg.size, device))
            for j, value in enumerate(result):
                Image.fromarray(decode(value)).save(out/'native'/f'common_{j:02d}_{name}_s{seed_index}.png')
        if name == 'post32':
            ids = [i for i, row in enumerate(rows) if row['quality_tier'] == 'highest_quality']
            for seed_index, seed in enumerate(SEEDS):
                for mode in ('text', 'joint'):
                    for start in range(0, len(ids), 12):
                        chosen = ids[start:start+12]
                        h = features['detail'][chosen].to(device)
                        refs = features['reference'][chosen].to(device) if mode == 'joint' else None
                        result = sample(model, h, torch.ones(h.shape[:2], dtype=torch.bool, device=device), refs,
                            seeds=[seed]*len(chosen), steps=32, initial_noise=noise_for([seed]*len(chosen), 32, device))
                        for j, i in enumerate(chosen):
                            Image.fromarray(decode(result[j])).save(out/'native'/f'{rows[i]["case_id"]}_post32_{mode}_detail_s{seed_index}.png')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('COMPLETE', name, round(time.monotonic()-started, 1), flush=True)
    atomic_json(out/f'progress_{progress_name}.json', {'phase': 'complete', 'seconds': time.monotonic()-started})


@torch.inference_mode()
def repeated_grid_check(out, device):
    """Duplicate positions AND noise; genuine32 forward should reproduce16 cells.

    This correlated-noise control is a structural test, not normal32 inference.
    """
    features = torch.load(out/'features.pt', weights_only=True)
    source = load_variant('pre16', device, out)
    cfg = copy.deepcopy(source.cfg)
    cfg.size = cfg.ref_size = 32
    repeated = PixelFlow(cfg).to(device).eval().requires_grad_(False)
    state = dict(source.state_dict())
    grid = source.position.reshape(1, 16, 16, cfg.width)
    state['position'] = grid.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(1, 1024, cfg.width)
    repeated.load_state_dict(state, strict=True)
    checks = []
    for mode in ('text', 'joint'):
        h = features['short'][:8].to(device)
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=device)
        refs = features['reference'][:8].to(device) if mode == 'joint' else None
        coarse = noise_for([SEEDS[0]]*len(h), 16, device)
        small = sample(source, h, mask, refs, seeds=[SEEDS[0]]*len(h), steps=32, initial_noise=coarse)
        large = sample(repeated, h, mask, refs, seeds=[SEEDS[0]]*len(h), steps=32,
                       initial_noise=coarse.repeat_interleave(2, 2).repeat_interleave(2, 3))
        expected = small.repeat_interleave(2, 2).repeat_interleave(2, 3)
        checks.append({'mode': mode, 'cases': len(h), 'max_absolute_difference': (large-expected).abs().max().item(),
                       'mean_absolute_difference': (large-expected).abs().mean().item(),
                       'close_at_1e-4': torch.allclose(large, expected, atol=1e-4, rtol=1e-4)})
    atomic_json(out/'repeated_grid_check.json', {'purpose': 'repeat-grid mathematical consistency only; correlated noise is not a standard32 prior', 'checks': checks})
    print('REPEATED GRID', checks, flush=True)


@torch.inference_mode()
def position_roundtrip(out, device):
    """Hold the16 grid fixed; change only the positional buffer amplitude."""
    features = torch.load(out/'features.pt', weights_only=True)
    initial = load_checkpoint(RUN/'training/initial_native32.pt')
    final = load_checkpoint(RUN/'training/latest.pt')
    if not torch.equal(initial['model']['position'], final['model']['position']):
        raise ValueError('Fixed position buffer changed during training')
    grid = initial['model']['position'].transpose(1, 2).reshape(1, 512, 32, 32)
    down = F.interpolate(grid, size=(16, 16), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
    del initial, final
    for original, name in [('pre16', 'pre16_downpos'), ('post16', 'post16_downpos')]:
        model = load_variant(original, device, out)
        model.position.copy_(down.to(device))
        h = features['common_text'].to(device)
        for seed_index, seed in enumerate(SEEDS):
            result = sample(model, h, torch.ones(h.shape[:2], dtype=torch.bool, device=device),
                seeds=[seed]*len(h), steps=32, initial_noise=noise_for([seed]*len(h), 16, device))
            for j, value in enumerate(result):
                Image.fromarray(decode(value)).save(out/'native'/f'common_{j:02d}_{name}_s{seed_index}.png')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(out/'position_roundtrip_check.json', {'complete': True,
        'purpose': 'All native16, same weights and Gaussian noise within each pair. Only source16 position vs downsampled interpolated32 position differs.',
        'fixed_position_buffer_unchanged_during_training': True, 'training_performed': False})
    print('POSITION ROUNDTRIP COMPLETE', flush=True)


@torch.inference_mode()
def detailed_generation(out, device):
    """Independent caption-only shard; same precision, seeds and model weights."""
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.18, device)
    rows = json.loads((out/'samples.json').read_text())
    features = torch.load(out/'features.pt', weights_only=True)
    model = load_variant('post32', device, out)
    ids = [i for i, row in enumerate(rows) if row['quality_tier'] == 'highest_quality']
    done = 0
    for seed_index, seed in enumerate(SEEDS):
        for mode in ('text', 'joint'):
            for start in range(0, len(ids), 12):
                chosen = [i for i in ids[start:start+12] if not (out/'native'/f'{rows[i]["case_id"]}_post32_{mode}_detail_s{seed_index}.png').exists()]
                if not chosen:
                    continue
                h = features['detail'][chosen].to(device)
                refs = features['reference'][chosen].to(device) if mode == 'joint' else None
                result = sample(model, h, torch.ones(h.shape[:2], dtype=torch.bool, device=device), refs,
                    seeds=[seed]*len(chosen), steps=32, initial_noise=noise_for([seed]*len(chosen), 32, device))
                for j, i in enumerate(chosen):
                    Image.fromarray(decode(result[j])).save(out/'native'/f'{rows[i]["case_id"]}_post32_{mode}_detail_s{seed_index}.png')
                done += len(chosen)
                atomic_json(out/'progress_detail.json', {'phase': 'running', 'done': done})
    atomic_json(out/'progress_detail.json', {'phase': 'complete', 'done': done})


def image_metrics(path, target):
    with Image.open(path) as im:
        generated = np.array(im.convert('RGBA').resize((32, 32), Image.Resampling.NEAREST)).astype(np.float32)/255
    truth = target.astype(np.float32)/255
    alpha = generated[..., 3] >= .5
    expected = truth[..., 3] >= .5
    intersection = np.logical_and(alpha, expected).sum()
    union = np.logical_or(alpha, expected).sum()
    premul = np.concatenate((generated[..., :3]*generated[..., 3:], generated[..., 3:]), -1)
    gt = np.concatenate((truth[..., :3]*truth[..., 3:], truth[..., 3:]), -1)
    visited = np.zeros_like(alpha)
    areas = []
    for y, x in zip(*np.nonzero(alpha)):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        area = 0
        while stack:
            yy, xx = stack.pop()
            area += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = yy+dy, xx+dx
                    if 0 <= ny < 32 and 0 <= nx < 32 and alpha[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        areas.append(area)
    foreground = int(alpha.sum())
    return {'premultiplied_mae': float(np.abs(premul-gt).mean()),
        'alpha_iou': float(intersection/union) if union else 1.,
        'largest_component_fraction': float(max(areas, default=0)/max(1, foreground)),
        'components_8_connected': len(areas), 'near_empty': foreground < 4}


def render(out):
    for group in ('native16', 'native32'):
        complete = json.loads((out/f'progress_{group}.json').read_text())['phase'] == 'complete'
        if group == 'native32' and not complete:
            complete = all((out/f'progress_{v}.json').exists() and json.loads((out/f'progress_{v}.json').read_text())['phase'] == 'complete'
                           for v in ('pre32', 'post32'))
        if not complete:
            raise ValueError('Generation is still running')
    rows = json.loads((out/'samples.json').read_text())
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    panels = []
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split]
        for page, offset in enumerate(range(0, len(selected), 8), 1):
            cases = selected[offset:offset+8]
            for mode in ('joint', 'text', 'reference', 'latest'):
                labels = (['HD reference', 'MC32 target', 'Before16', 'Before32', 'After32', 'After16*'] if mode != 'latest'
                          else ['HD reference', 'MC32 target', 'Text only', 'Reference only', 'Text + ref', 'Joint seed B'])
                canvas = Image.new('RGB', (938, 66+164*len(cases)), '#111827')
                draw = ImageDraw.Draw(canvas)
                title = f'{split.upper()} | {mode} | matched noise,32-step Heun | *After16 is regrid diagnostic'
                draw.text((14, 10), title, font=font, fill='white')
                for c, label in enumerate(labels):
                    draw.text((14+c*154, 34), label, font=font, fill='#cbd5e1')
                for j, row in enumerate(cases):
                    ident = row['case_id']
                    y = 62+j*164
                    paths = [out/'inputs'/f'{ident}_reference.png', out/'inputs'/f'{ident}_target.png']
                    if mode != 'latest':
                        paths += [out/'native'/f'{ident}_{v}_{mode}_s0.png' for v in VARIANTS]
                    else:
                        detail = '_detail' if row['quality_tier'] == 'highest_quality' else ''
                        paths += [out/'native'/f'{ident}_post32_text{detail}_s0.png', out/'native'/f'{ident}_post32_reference_s0.png',
                                  out/'native'/f'{ident}_post32_joint{detail}_s0.png', out/'native'/f'{ident}_post32_joint{detail}_s1.png']
                    for c, path in enumerate(paths):
                        with Image.open(path) as im:
                            canvas.paste(thumbnail(im, 128, native=c!=0), (14+c*154, y))
                    draw.text((14, y+132), f'{ident} | {row["captions"][0]}', font=font, fill='white')
                    draw.text((14, y+147), f'{row["quality_tier"]} | {row["source_family"]}', font=font, fill='#aebbd0')
                name = f'{split}_{mode}_{page:02d}.png'
                canvas.save(out/name)
                panels.append(name)
    features = torch.load(out/'features.pt', weights_only=True)
    canvas = Image.new('RGB', (654, 66+164*len(features['common_prompts'])), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), 'Common prompts | Before16 / Before32 / After32 / After16*', font=font, fill='white')
    for c, label in enumerate(('Before16', 'Before32', 'After32', 'After16*')):
        draw.text((14+c*160, 34), label, font=font, fill='#cbd5e1')
    for j, prompt in enumerate(features['common_prompts']):
        y = 62+j*164
        for c, variant in enumerate(VARIANTS):
            with Image.open(out/'native'/f'common_{j:02d}_{variant}_s0.png') as im:
                canvas.paste(thumbnail(im, 128, native=True), (14+c*160, y))
        draw.text((14, y+132), prompt, font=font, fill='white')
    canvas.save(out/'common_resolution.png')
    panels.append('common_resolution.png')
    if (out/'position_roundtrip_check.json').exists():
        canvas = Image.new('RGB', (654, 66+164*len(features['common_prompts'])), '#111827')
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), 'All native16 | change only positions within each weight pair', font=font, fill='white')
        variants = ('pre16', 'pre16_downpos', 'post16', 'post16_downpos')
        for c, label in enumerate(('Old/original pos', 'Old/roundtrip pos', 'Final/original pos', 'Final/roundtrip pos')):
            draw.text((14+c*160, 34), label, font=font, fill='#cbd5e1')
        for j, prompt in enumerate(features['common_prompts']):
            y = 62+j*164
            for c, variant in enumerate(variants):
                with Image.open(out/'native'/f'common_{j:02d}_{variant}_s0.png') as im:
                    canvas.paste(thumbnail(im, 128, native=True), (14+c*160, y))
            draw.text((14, y+132), prompt, font=font, fill='white')
        canvas.save(out/'position_roundtrip.png')
        panels.append('position_roundtrip.png')
    metrics = []
    for row in rows:
        with Image.open(row['target_file']) as im:
            target = np.array(im.convert('RGBA'))
        for variant in VARIANTS:
            for mode in ('text', 'reference', 'joint'):
                for seed_index in range(2):
                    path = out/'native'/f'{row["case_id"]}_{variant}_{mode}_s{seed_index}.png'
                    with Image.open(path) as im:
                        if im.size != ((16, 16) if variant.endswith('16') else (32, 32)) or im.mode != 'RGBA':
                            raise ValueError('Unexpected native output')
                    metrics.append({'case_id': row['case_id'], 'split': row['split'], 'quality_tier': row['quality_tier'],
                        'variant': variant, 'mode': mode, 'seed_index': seed_index, **image_metrics(path, target)})
    summary = []
    for split in ('train', 'val'):
        for variant in VARIANTS:
            for mode in ('text', 'reference', 'joint'):
                group = [m for m in metrics if (m['split'], m['variant'], m['mode']) == (split, variant, mode)]
                summary.append({'split': split, 'variant': variant, 'mode': mode, 'outputs': len(group),
                    **{key: sum(float(m[key]) for m in group)/len(group) for key in ('premultiplied_mae', 'alpha_iou', 'largest_component_fraction', 'near_empty')}})
    atomic_json(out/'metrics.json', {'per_output': metrics, 'summary': summary,
        'caveat': 'Alpha/target metrics are diagnostic proxies; they do not establish MC style or decoration fidelity.16 outputs are nearest-upscaled only for common32 metric space.'})
    native_files = []
    for path in sorted((out/'native').glob('*.png')):
        expected = 16 if re.search(r'_(?:pre16|post16)_', path.name) else 32
        with Image.open(path) as im:
            if im.mode != 'RGBA' or im.size != (expected, expected):
                raise ValueError('Invalid native output: '+str(path))
        native_files.append({'path': path.name, 'native_size': expected, 'sha256': file_hash(path)})
    atomic_json(out/'generation_manifest.json', native_files)
    atomic_json(out/'verification.json', {'complete': True, 'primary_outputs': len(metrics),
        'all_native_pngs': len(list((out/'native').glob('*.png'))), 'source_learned_weights_match_initial32':
        json.loads((out/'model_pre16.json').read_text())['learned_weights_sha256'] == json.loads((out/'model_pre32.json').read_text())['learned_weights_sha256'],
        'post16_learned_weights_match_post32': json.loads((out/'model_post16.json').read_text())['learned_weights_sha256'] == json.loads((out/'model_post32.json').read_text())['learned_weights_sha256'],
        'training_performed': False, 'panels': panels})
    body = '<!doctype html><meta charset="utf-8"><title>Resolution validation</title><style>body{background:#111827;color:white;font:16px sans-serif;max-width:1100px;margin:24px auto}img{max-width:100%}a{color:#93c5fd}section{margin:36px 0}.controls{position:sticky;top:0;background:#1f2937;padding:16px;z-index:2}select{padding:8px;margin:0 20px 0 8px}.grid{display:grid;grid-template-columns:repeat(6,minmax(90px,1fr));gap:10px}.grid img{width:128px;height:128px;object-fit:contain;image-rendering:pixelated;background:#edf0f4}.grid img.hd{image-rendering:auto}article{margin:20px 0;padding:14px;background:#1f2937;border-radius:10px}h3{font-size:15px}</style>'
    body += '<h1>Expanded validation and resolution diagnostics</h1><p>56 fixed targets, two Gaussian seeds. Before16/Before32 share learned weights; After32/After16 share learned weights. After16 is an inference diagnostic, not a model retrained at16.</p>'
    body += '<div class="controls"><label>Split<select id="split"><option value="val">Validation</option><option value="train">Train</option><option value="all">All</option></select></label><label>Mode<select id="mode"><option value="joint">Text + reference</option><option value="text">Text only</option><option value="reference">Reference only</option></select></label><label>Seed<select id="seed"><option value="0">A</option><option value="1">B</option></select></label></div><div id="samples"></div>'
    entries = json.dumps([{'id': r['case_id'], 'split': r['split'], 'caption': r['captions'][0], 'quality': r['quality_tier']} for r in rows]).replace('<', '\\u003c')
    body += '<script>const rows='+entries+';function draw(){const container=document.getElementById("samples");container.replaceChildren();const split=document.getElementById("split").value,mode=document.getElementById("mode").value,seed=document.getElementById("seed").value;for(const row of rows){if(split!=="all"&&row.split!==split)continue;const card=document.createElement("article"),title=document.createElement("h3");title.textContent=row.id+" | "+row.caption+" | "+row.quality;card.append(title);const grid=document.createElement("div");grid.className="grid";const variants=["pre16","pre32","post32","post16"],labels=["HD reference","MC target","Before16","Before32","After32","After16 (diagnostic)"];const paths=["inputs/"+row.id+"_reference.png","inputs/"+row.id+"_target.png",...variants.map(v=>"native/"+row.id+"_"+v+"_"+mode+"_s"+seed+".png")];paths.forEach((path,i)=>{const cell=document.createElement("div"),label=document.createElement("p"),image=document.createElement("img");label.textContent=labels[i];image.src=path;image.alt=labels[i];image.loading="lazy";if(i===0)image.className="hd";cell.append(label,image);grid.append(cell)});card.append(grid);container.append(card)}}for(const id of ["split","mode","seed"])document.getElementById(id).addEventListener("change",draw);draw();</script>'
    body += '<h2>All static panels</h2>'
    for name in ['common_resolution.png']+[p for p in panels if p != 'common_resolution.png']:
        body += f'<details><summary>{html.escape(name)}</summary><section><a href="{name}"><img loading="lazy" src="{name}"></a></section></details>'
    (out/'index.html').write_text(body)
    print('RENDERED', len(metrics), 'primary outputs;', len(panels), 'panels', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'generate', 'repeat-check', 'position-check', 'detail', 'render'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--group', choices=['native16', 'native32'], default='native32')
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
    elif args.stage == 'generate':
        generate(args.out, args.group, args.device, args.variant)
    elif args.stage == 'render':
        render(args.out)
    elif args.stage == 'detail':
        detailed_generation(args.out, args.device)
    else:
        torch.set_num_threads(4)
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(.18, args.device)
        if args.stage == 'position-check':
            position_roundtrip(args.out, args.device)
        else:
            repeated_grid_check(args.out, args.device)
