"""Sample real train/test captions and generate an auditable qualitative figure set."""
import argparse
from collections import Counter, defaultdict
import gc
import html
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import content_hash, decode
from mcgen.model import PixelFlow
from mcgen.resource_text import extract_resource_text
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts

RUN = Path('outputs/qwen_vl_embedding_large_20261103')
CHECKPOINT = Path('checkpoints/mcgen-step4128.pt')
CATEGORIES = {
    'weapons': set('sword dagger spear bow crossbow trident mace katana rapier'.split()),
    'tools': set('pickaxe shovel axe hoe drill saw wrench sickle hammer'.split()),
    'equipment': set('helmet chestplate leggings boots shield hood robe gauntlet'.split()),
    'gems': set('gem crystal jewel pearl shard diamond ruby sapphire emerald amethyst opal quartz'.split()),
    'materials': set('ingot nugget dust powder plate gear wire rod alloy'.split()),
    'magic': set('wand staff tome spell rune scroll book sigil charm amulet'.split()),
    'food': set('apple bread cookie cake pie stew soup beef carrot berry berries cheese mushroom sandwich'.split()),
    'objects': set('potion bottle bucket key coin ring pendant lantern compass clock totem boat minecart'.split()),
}
PARTS = set('broken blade handle pommel head grip guard binding limb overlay underlay empty frame anim model texture render gui hud placeholder toolhead toolrod toolwrap part parts'.split())
TECHNICAL = set('tinker tinkers tinkersadvanced tconstruct ticex texture item textures blueprint spawn egg new old temp debug variant alt'.split())
TECHNICAL_FORMS = {form + suffix for forms in CATEGORIES.values() for form in forms
                   for suffix in ('new', 'old', 'temp', 'alt')}
ATTRIBUTES = set('red blue green purple yellow orange cyan pink black white gray grey dark light lime magenta copper iron gold golden silver bronze steel diamond emerald ruby sapphire amethyst quartz obsidian netherite prismarine lapis wooden wood stone basalt clay bone leather wool silk ancient arcane enchanted magical magic celestial solar lunar moon sun star soul spirit void ender blood frost ice fire flame blazing burning thunder storm electric lightning nature earth air wind water ocean sea deep holy sacred cursed shadow phantom spectral ghost toxic venom poison radioactive molten refined reinforced hardened polished plated pure raw weathered royal crimson azure scarlet violet verdant luminous glowing bright heavy tin lead zinc tungsten titanium platinum brass honey salt berry mushroom vegetable wheat milk cheese chocolate cooked beef chicken fish bread peanut butter quantum dragon draconic scarab phoenix echo frozen glacial astral cosmic'.split())


def classify(prompt):
    if not re.fullmatch(r'[a-z]+(?: [a-z]+){1,5}', prompt):
        return None
    words = prompt.split()
    if not 10 <= len(prompt) <= 48 or max(map(len, words)) > 16 or min(map(len, words)) < 2:
        return None
    if set(words) & (PARTS | TECHNICAL):
        return None
    if any(len(word) == 2 and word not in {'of', 'in', 'on', 'at', 'to', 'by', 'or', 'an', 'as', 'up'} for word in words):
        return None
    if set(words) & TECHNICAL_FORMS:
        return None
    if not set(words) & ATTRIBUTES:
        return None
    terminal = next((category for category, forms in CATEGORIES.items() if words[-1] in forms), None)
    if terminal is None:
        return None
    # Item forms outrank material modifiers: "helmet emerald" is equipment.
    primary = [category for category, forms in CATEGORIES.items()
               if category not in ('gems', 'materials') and set(words) & forms]
    if len(primary) > 1:
        return None
    if primary:
        category = primary[0]
        return (category, words[-1]) if category == terminal else None
    for word in reversed(words):
        for category, forms in CATEGORIES.items():
            if word in forms:
                return category, word
    return None


def prepare(out, selection_seed):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'targets').mkdir()
    train = []
    train_hashes, train_groups, train_lineages, train_prompts = set(), set(), set(), set()
    for line in (RUN / 'catalog/samples.jsonl').open():
        row = json.loads(line)
        if row['split'] != 'train':
            continue
        train.append(row)
        train_hashes.add(row['target_hash'])
        train_groups.add(row['split_group'])
        train_lineages.update(row.get('source_lineages', [row['lineage']]))
        train_prompts.add(row['prompt'])
    metadata = pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',
                             columns=['file_name', 'mod_slug']).to_pydict()
    original_test, target_types = {}, defaultdict(set)
    for line in Path('data/catalogs/native16_v3/catalog/samples.jsonl').open():
        row = json.loads(line)
        if row['split'] != 'test':
            continue
        target_types[row['target_hash']].add(row['asset_type'])
        if row['asset_type'] == 'item' and row['id'].startswith('legacy-text:'):
            original_test.setdefault(row['target_hash'], row)
    test = []
    for digest, row in original_test.items():
        if target_types[digest] != {'item'} or digest in train_hashes:
            continue
        if row['split_group'] in train_groups or train_lineages.intersection(row.get('source_lineages', [row['lineage']])):
            continue
        name = metadata['file_name'][row['source_row']]
        mod = metadata['mod_slug'][row['source_row']]
        policy = extract_resource_text(name, mod)
        if policy['decision'] != 'keep' or policy['text'] in train_prompts:
            continue
        test.append({**row, 'prompt': policy['text'], 'source_filename': name, 'source_mod': mod})
    rng = random.Random(selection_seed)
    maps, selected, selected_hashes, selected_prompts = {}, [], set(), set()
    counts = {}
    for split, pool in [('train', train), ('test', test)]:
        candidates = defaultdict(list)
        for row in sorted(pool, key=lambda r: r['target_hash']):
            label = classify(row['prompt'])
            if label and row.get('alpha_mode') != 'opaque':
                candidates[label[0]].append((row, label[1]))
        counts[split] = {key: len(candidates[key]) for key in CATEGORIES}
        projects = Counter()
        for category in CATEGORIES:
            rows = candidates[category]
            rng.shuffle(rows)
            forms, accepted = set(), []
            for slot in range(2):
                # At least one longer caption per class, then favor a new item form.
                ordered = sorted(rows, key=lambda p: (len(p[0]['prompt'].split()) < 3 if slot == 0 else p[1] in forms))
                for row, form in ordered:
                    if row['target_hash'] in selected_hashes or row['prompt'] in selected_prompts:
                        continue
                    if projects[row['split_group']] >= 2:
                        continue
                    spec = row['target']
                    key = (spec['path'], tuple(spec['shape']))
                    if key not in maps:
                        maps[key] = np.memmap(key[0], mode='r', dtype='uint8', shape=key[1])
                    pixels = np.array(maps[key][spec['index']])
                    assert content_hash(pixels) == row['target_hash']
                    visible = int(np.count_nonzero(pixels[..., 3] > 8))
                    if not 24 <= visible <= 236:
                        continue
                    i = len(selected)
                    Image.fromarray(pixels).save(out / 'targets' / f'{i:02d}.png')
                    selected.append({'split': split, 'category': category, 'prompt': row['prompt'],
                                     'source_filename': row['source_filename'], 'source_mod': row.get('source_mod', ''),
                                     'source_row': row['source_row'], 'target_hash': row['target_hash'],
                                     'split_group': row['split_group'], 'lineage': row['lineage'],
                                     'visible_pixels': visible, 'sample_index': i,
                                     'prompt_seen_in_training': row['prompt'] in train_prompts})
                    selected_hashes.add(row['target_hash'])
                    selected_prompts.add(row['prompt'])
                    projects[row['split_group']] += 1
                    forms.add(form)
                    accepted.append(row)
                    break
                else:
                    raise ValueError(f'Not enough eligible samples: {split}/{category}')
            assert len(accepted) == 2
    assert len(selected) == 32
    assert all(not r['prompt_seen_in_training'] for r in selected if r['split'] == 'test')
    atomic_json(out / 'samples.json', selected)
    atomic_json(out / 'protocol.json', {
        'selection_seed': selection_seed, 'generation_seed': 20261112,
        'counts': {'train': 16, 'test': 16}, 'candidate_counts': counts,
        'sampling': 'Two samples in each of eight caption-defined categories per split; one longer caption preferred, different forms preferred, at most two targets per project. Fixed random seed; no filtering by generated quality.',
        'eligibility': '2-6 lowercase words,10-48 characters; explicit terminal item form and at least one word in the declared material/color/theme vocabulary; single-letter/component/technical markers and competing item forms excluded; nonopaque items with24-236 visible pixels. These are display-caption filters, not changes to training supervision.',
        'train_source': str(RUN / 'catalog/samples.jsonl'),
        'test_source': 'data/catalogs/native16_v3/catalog/samples.jsonl',
        'test_policy': 'Original test split, unique item-only targets; no training target/group/lineage overlap; filename policy unchanged; exact prompt strings also absent from training captions.',
        'test_exposure': 'The32 selected examples are now viewed development/showcase examples. They are not an untouched final test benchmark. No training catalog or split is modified.',
        'interpretation': 'Illustrative stratified showcase, not a population-weighted quality estimate. Target images are stored locally for provenance, never substituted for generated outputs.',
        'steps': 32, 'solver': 'heun', 'text_scale': 2, 'precision': 'float32',
    })
    for row in selected:
        print(row['sample_index'], row['split'], row['category'], row['prompt'], flush=True)
    print('PREPARED', out, flush=True)


def generate(out, requested_device):
    torch.set_num_threads(4)
    if requested_device == 'auto':
        available = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
        if not available or max(available)[0] < 30 * 1024**3:
            raise RuntimeError('No GPU with30GiB free memory; no other workload was stopped.')
        device = 'cuda:' + str(max(available)[1])
    else:
        device = requested_device
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.55, device)
    samples = json.loads((out / 'samples.json').read_text())
    state = load_checkpoint(CHECKPOINT)
    assert state['step'] == 4128
    text_identity = state['text_identity']
    cfg = Experiment.from_dict(state['config'])
    keys = conditioned_prompts([r['prompt'] for r in samples], cfg)
    features_path = out / 'text_features.pt'
    if features_path.exists():
        stored = torch.load(features_path, weights_only=True, map_location='cpu')
        assert stored['identity'] == state['text_identity'] and stored['prompts'] == keys
        encoded = stored['features']
    else:
        cache = DiskTextEncoder(RUN / 'features', state['text_identity'])
        missing = list(dict.fromkeys(p for p in keys if p not in cache.index))
        online = TextEncoder(cfg.text, cfg.model, device) if missing else None
        if online is not None:
            assert online.identity == state['text_identity']
        encoded = {}
        for provider, subset in [(cache, list(dict.fromkeys(p for p in keys if p in cache.index))), (online, missing)]:
            for start in range(0, len(subset), 16):
                part = subset[start:start + 16]
                hidden, mask = provider.encode(part, 'cpu')
                for i, prompt in enumerate(part):
                    encoded[prompt] = (hidden[i].clone(), mask[i].clone())
        del provider, online
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        torch.save({'identity': state['text_identity'], 'prompts': keys, 'features': encoded}, features_path)
        print('Encoded', len(keys), 'captions;', len(missing), 'new embeddings', flush=True)
    model = PixelFlow(cfg.model).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    del state
    (out / 'images').mkdir(exist_ok=True)
    for start in range(0, len(samples), 8):
        batch_keys = keys[start:start + 8]
        hidden = torch.stack([encoded[p][0] for p in batch_keys]).to(device)
        mask = torch.stack([encoded[p][1] for p in batch_keys]).to(device)
        with torch.inference_mode():
            values = sample(model, hidden, mask, seeds=[20261112] * len(batch_keys),
                            steps=32, text_scale=2, solver='heun')
        assert torch.isfinite(values).all()
        for i, value in enumerate(values):
            Image.fromarray(decode(value)).save(out / 'images' / f'{start + i:02d}.png')
        print('Generated', start + len(batch_keys), len(samples), device, flush=True)
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol.update(checkpoint_sha256=file_hash(CHECKPOINT), checkpoint_step=4128,
                    text_identity=text_identity,
                    device=device)
    atomic_json(out / 'protocol.json', protocol)
    parts = ['<!doctype html><meta charset="utf-8"><title>Dataset caption showcase</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd}img{width:128px;height:128px;image-rendering:pixelated;background:#eee}</style><h1>训练集与测试集标签生图 · step4128</h1><p>原图只在本地作为来源对照；右列才是模型输出。分层抽样不是总体质量评分。</p><table><tr><th>划分 / 类别 / 提示词</th><th>数据原图</th><th>生成结果</th></tr>']
    for row in samples:
        i = row['sample_index']
        parts.append(f'<tr><td>{row["split"]} / {row["category"]}<br>{html.escape(row["prompt"])}</td><td><img src="targets/{i:02d}.png"></td><td><img src="images/{i:02d}.png"></td></tr>')
    (out / 'review.html').write_text(''.join(parts) + '</table>')
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection-seed', type=int, default=20261104)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out, args.selection_seed)
    else:
        generate(args.out, args.device)
