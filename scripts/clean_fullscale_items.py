"""Reversible, conservative item-only cleaning; never change parent splits or pixels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import textwrap

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from mcgen.catalog import atomic_json, file_hash, write_catalog
from mcgen.images import content_hash
from mcgen.semantic_captions import filename_words

ROOT = Path('outputs/fullscale_clean_v1_20261102')
PARENT = Path('data/catalogs/native16_v3/catalog')
CATEGORIES = 'pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot gem apple book bow'.split()


def flags(name, mod='', refined=False):
    words = set(w.lower() for w in filename_words(name))
    result = []
    if not words:
        result.append('empty_filename_text')
    # Quarantine ambiguous uses, not a claim that all matches are invalid images.
    layer_words = set('overlay underlay hud cursor slot placeholder outline'.split())
    if not refined: layer_words.add('gui')
    if words & layer_words:
        result.append('possible_render_layer_or_ui')
    if words & set('blade binding handle toolrod toolhead'.split()) or (
        'head' in words and words & set('pickaxe axe shovel hammer hoe mattock scythe cleaver'.split())
    ) or ('rod' in words and words & {'tool', 'tough'}) or (
        words & {'part', 'parts'} and words & set(CATEGORIES + ['tool', 'armor'])
    ):
        result.append('possible_assembly_component')
    if refined:
        context = (name + ' ' + (mod or '')).lower()
        tinker = any(x in context for x in ('tinker', 'tconstruct', 'materialis', 'construct', 'tcintegrations'))
        if 'blade' in words and not tinker and not words & {'binding', 'handle', 'toolrod', 'toolhead', 'small', 'broad', 'broken'}:
            result = [x for x in result if x != 'possible_assembly_component']
        if words & {'crossguard', 'embellishment', 'embellishments'} or (tinker and words & {'front', 'back', 'broken'}):
            result.append('possible_assembly_component')
        if tinker and words & {'limb', 'plating'}:
            result.append('possible_assembly_component')
        if {'chem', 'icon'} <= words:
            result.append('possible_render_layer_or_ui')
        if ('3d' in name.lower() and any(w.isdigit() for w in words)) or words & {'texturemap', 'uvmap'}:
            result.append('possible_model_texture')
        if {'unknown', 'item'} <= words or {'missing', 'texture'} <= words:
            result.append('placeholder_asset')
    return sorted(set(result))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    parent = json.loads((PARENT / 'catalog.json').read_text())
    assert file_hash(PARENT / 'samples.jsonl') == parent['manifest_sha256']
    names = pq.read_table('data/build/mc_text2image32_wl/metadata.parquet', columns=['file_name', 'mod_slug']).to_pydict()
    records = {}; types = defaultdict(set); aliases = defaultdict(set); lineages = defaultdict(set)
    owners = {}; source_rows = Counter()
    for line in (PARENT / 'samples.jsonl').open():
        r = json.loads(line); h = r['target_hash']; split = r['split']
        assert owners.setdefault(h, split) == split
        if split == 'test':
            continue  # Test remains frozen, not inspected or used to choose rules.
        types[h].add(r['asset_type'])
        if r['asset_type'] != 'item':
            continue
        source_rows[split] += 1
        i = r['source_row']; assert 0 <= i < len(names['file_name'])
        aliases[h].add((names['file_name'][i] or '', names['mod_slug'][i] or ''))
        lineages[h].add(r['lineage'])
        # Prefer original text row to generated-reference pair; deterministic thereafter.
        rank = (not r['id'].startswith('legacy-text:'), r['id'])
        if h not in records or rank < records[h][0]: records[h] = (rank, r)
    prior = json.load(open('outputs/text_expansion2k_20261102/full_visual_review.json'))
    candidates = {r['id']: r for r in json.load(open('outputs/text_expansion2k_20261102/candidates.json'))}
    known = {candidates[r['id']]['target_hash']: r['reason'] for r in prior['excluded']
             if r['reason'] in ('partial_asset_or_overlay', 'ui_mold_background_or_texture')}
    maps = {}; counts = Counter()
    with (ROOT / 'inventory.jsonl').open('w') as out:
        for h, (_, r) in records.items():
            spec = r['target']; p = spec['path']
            if p not in maps:
                assert file_hash(p) == parent['sources'][p]['sha256']
                maps[p] = np.memmap(p, mode='r', dtype='uint8', shape=tuple(spec['shape']))
            a = np.asarray(maps[p][spec['index']]); assert content_hash(a) == h
            name = names['file_name'][r['source_row']] or ''
            reasons = flags(name)
            if 'tile' in types[h]: reasons.append('item_tile_type_conflict')
            if 'other' in types[h]: reasons.append('item_other_type_conflict')
            if not np.any(a[..., 3]): reasons.append('fully_transparent')
            if h in known: reasons.append('prior_visual_' + known[h])
            words = [w.lower() for w in filename_words(name)]
            r = {**r, 'reference': None, 'generator': 'none', 'source_filename': name,
                 'source_mod': names['mod_slug'][r['source_row']], 'filename_text': ' '.join(filename_words(name)),
                 'source_aliases': sorted(aliases[h]), 'source_lineages': sorted(lineages[h]),
                 'source_asset_types': sorted(types[h]), 'cleaning_flags': reasons,
                 'category_15': next((c for c in CATEGORIES if c in words), 'other'),
                 'visible_pixels': int(np.count_nonzero(a[..., 3])),
                 'cleaning_decision': 'quarantine' if reasons else 'keep'}
            out.write(json.dumps(r, ensure_ascii=False) + '\n'); counts[r['split'] + '/' + r['cleaning_decision']] += 1
    atomic_json(ROOT / 'preparation.json', {'parent_fingerprint': parent['fingerprint'], 'source_item_rows': dict(source_rows),
        'counts': dict(counts), 'inventory_sha256': file_hash(ROOT / 'inventory.jsonl'), 'test_untouched': True,
        'policy': 'Quarantine type conflicts, explicit layer/component filename patterns and previously visually rejected incomplete assets; preserve numeric variants, abstract names, unusual shapes, opaque and continuous-alpha items. No source/style cap. Not exhaustive visual certification.',
        'script_sha256': file_hash(__file__)})
    print(dict(counts), flush=True)


def sheets():
    pools = defaultdict(list)
    for line in (ROOT / 'inventory.jsonl').open():
        r = json.loads(line)
        if r['split'] != 'train': continue
        if r['cleaning_decision'] == 'keep':
            pools['keep_random'].append(r)
            if r['category_15'] == 'other': pools['keep_longtail'].append(r)
            if r['alpha_mode'] == 'opaque': pools['keep_opaque'].append(r)
        else:
            for reason in r['cleaning_flags']: pools[reason].append(r)
    out = ROOT / 'review'; out.mkdir(exist_ok=True); selected = {}; maps = {}
    for group, pool in sorted(pools.items()):
        random.Random(20261103).shuffle(pool); sample = pool[:48]; selected[group] = []
        sheet = Image.new('RGB', (1200, 1050), '#eeeeee'); draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), f'{group}: {len(sample)} / {len(pool)} (train only)', fill='black')
        for i, r in enumerate(sample):
            x = (i % 8)*150; y = (i//8)*170 + 28; spec = r['target']
            if spec['path'] not in maps: maps[spec['path']] = np.memmap(spec['path'], mode='r', dtype='uint8', shape=tuple(spec['shape']))
            im = Image.fromarray(np.array(maps[spec['path']][spec['index']])).resize((96, 96), Image.Resampling.NEAREST)
            bg = Image.new('RGB', (96, 96), '#999999'); bg.paste(im, mask=im.getchannel('A')); sheet.paste(bg, (x+20,y))
            label = f'{i}: {r["source_filename"]}'
            draw.multiline_text((x+3,y+98), '\n'.join(textwrap.wrap(label, 23)[:4]), fill='black')
            selected[group].append({'index': i, 'target_hash': r['target_hash'], 'id': r['id'], 'filename': r['source_filename']})
        sheet.save(out / (group + '.png'))
    atomic_json(out / 'samples.json', selected)
    print({k: len(v) for k,v in selected.items()}, flush=True)


def build():
    if (ROOT / 'catalog').exists() or (ROOT / 'decisions.jsonl').exists():
        raise ValueError('Use a fresh output root; never overwrite frozen cleaning decisions.')
    review = json.load(open(ROOT / 'review' / 'decisions.json'))
    overrides = review.get('overrides', {})
    descriptions = {r['target_hash']: r for r in json.load(open('outputs/text_expansion2k_20261102/cases.json'))}
    records = []; counts = Counter(); reasons = Counter(); before = Counter(); after = Counter()
    projects = defaultdict(set); alpha = Counter(); names = Counter(); supervision = Counter(); memberships = []
    seen = set(); before_splits = {}; after_splits = {}
    with (ROOT / 'decisions.jsonl').open('w') as decisions:
        for line in (ROOT / 'inventory.jsonl').open():
            r = json.loads(line); h = r['target_hash']; split = r['split']; assert h not in seen; seen.add(h)
            before_splits[h] = split
            structural = [f for f in r['cleaning_flags'] if f not in ('empty_filename_text', 'possible_render_layer_or_ui', 'possible_assembly_component')]
            r['cleaning_flags'] = sorted(set(structural + flags(r['source_filename'], r['source_mod'], refined=True)))
            if r['alpha_mode'] == 'opaque': r['cleaning_flags'].append('opaque_asset_requires_review')
            decision = overrides.get(h, {}).get('decision', 'quarantine' if r['cleaning_flags'] else 'keep')
            assert decision in ('keep', 'quarantine')
            counts[split + '/' + decision] += 1
            if split == 'train': before[r['category_15']] += 1
            decisions.write(json.dumps({'target_hash': h, 'split': split, 'decision': decision,
                'flags': r['cleaning_flags'], 'override': overrides.get(h), 'source_filename': r['source_filename']})+'\n')
            if decision != 'keep':
                if split == 'train': reasons.update(r['cleaning_flags'] or ['visual_sample_rejection'])
                continue
            assert r['filename_text'].strip() and '_' not in r['filename_text']
            after_splits[h] = split
            if split == 'train':
                after[r['category_15']] += 1; alpha[r['alpha_mode']] += 1; names[r['filename_text'].lower()] += 1
                for project in r['source_lineages']: projects[project].add(h)
            base = {**r, 'cleaning_decision': 'keep', 'prompt': r['filename_text'], 'caption_variant': 'short',
                    'id': 'clean:' + h + ':short', 'caption_supervision': 'filename_weak_label'}
            records.append(base)
            desc = descriptions.get(h, {})
            long = desc.get('descriptive_text') if desc.get('long_caption_eligible') else None
            if long and long != r['filename_text']:
                assert '_' not in long
                records.append({**base, 'id': 'clean:' + h + ':long', 'prompt': long, 'caption_variant': 'long',
                                'caption_supervision': 'retained_experimental_automatic_weak_label'})
            supervision[split + ('/dual' if long and long != r['filename_text'] else '/filename_only')] += 1
    assert set(overrides) <= seen
    assert all(before_splits[h] == s for h,s in after_splits.items())
    prep = json.load(open(ROOT / 'preparation.json'))
    assert file_hash(ROOT / 'inventory.jsonl') == prep['inventory_sha256']
    catalog = write_catalog(ROOT / 'catalog', records, {'native_size': 16, 'parent_fingerprint': prep['parent_fingerprint'],
        'cleaning_policy': prep['policy'] + ' Refined rules remove generic gui/blade false positives, add assembly/model patterns and quarantine unreviewed opaque assets after visual inspection found substantial surface-texture contamination.', 'inventory_sha256': prep['inventory_sha256'],
        'build_script_sha256': file_hash(__file__),
        'decisions_sha256': file_hash(ROOT / 'decisions.jsonl'), 'visual_review_sha256': file_hash(ROOT / 'review/decisions.json'),
        'caption_source_sha256': file_hash('outputs/text_expansion2k_20261102/cases.json'),
        'test_untouched': True, 'sampling': 'Uniform target, uniform eligible caption within target; no balancing introduced.'})
    ps = sorted(projects.items(), key=lambda x: len(x[1]), reverse=True)
    n = counts['train/keep']
    report = {'counts': dict(counts), 'train_categories_before': dict(before), 'train_categories_after': dict(after),
              'train_quarantine_reasons_nonexclusive': dict(reasons), 'train_alpha': dict(alpha),
              'train_projects': len(projects), 'top10_project_union_fraction': len(set().union(*(v for _,v in ps[:10])))/n,
              'top50_project_union_fraction': len(set().union(*(v for _,v in ps[:50])))/n,
              'supervision': dict(supervision), 'expected_description_fraction_given_text': supervision['train/dual']/n/2,
              'train_unique_names': len(names), 'train_images_sharing_name': sum(v for v in names.values() if v>1),
              'catalog_fingerprint': catalog['fingerprint'], 'audit': catalog['audit'],
              'test_untouched': True, 'limits': 'Rule-cleaned plus train-only stratified visual spot checks, not full visual review. Categories are filename heuristics. Quarantine is recoverable, not a verdict that each image is invalid.'}
    atomic_json(ROOT / 'report.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'sheets', 'build'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(); ROOT = args.root
    globals()[args.stage]()
