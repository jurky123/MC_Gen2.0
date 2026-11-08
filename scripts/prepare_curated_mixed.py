"""Verify the HQ delivery and snapshot reviewed legacy pairs into a native32 run."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image
import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.curated_training import target_captions
from mcgen.images import content_hash, native_grid, rgba
from mcgen.pair_selection import pair_id


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_image(root, row):
    path = root/row['path']
    if file_hash(path) != row['sha256']:
        raise ValueError('Image bytes differ: '+str(path))
    with Image.open(path) as im:
        array = np.array(im.convert('RGBA'))
        if im.size != (row['width'], row['height']):
            raise ValueError('Image dimensions differ')
    if hashlib.sha256(array.tobytes()).hexdigest() != row['rgba_sha256']:
        raise ValueError('Image pixels differ: '+str(path))
    return row['path']


def prepare(root, bundle, legacy):
    if (root/'initialization.json').exists() or (root/'config.yaml').exists():
        raise ValueError('Use a new directory; a prepared experiment is never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    processed = Path('data/processed/high_quality/MC_Gen_20261106')
    processed.mkdir(parents=True, exist_ok=True)
    targets = Path('data/processed/reference_mixed_20261107/native32')
    targets.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    images = read_jsonl(bundle/'metadata/images.jsonl')
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(lambda r: verify_image(bundle, r), images))
    image_map = {r['path']: r for r in images}
    new_rows = read_jsonl(bundle/'metadata/pairs_main_1000.jsonl')
    if len(new_rows) != 1000 or len({r['sample_id'] for r in new_rows}) != 1000:
        raise ValueError('Expected exactly 1000 distinct primary pairs')
    if any(r['group'] != 'main_1000' or r['version'] != 'current'
           or r.get('original_training_approved') is False for r in new_rows):
        raise ValueError('Nonprimary/unapproved sample in primary manifest')
    main_provenance = {r['id']: r for r in json.loads((bundle/'source_manifests/mcgen2_items_500/source_provenance.json').read_text())}
    gear_provenance = {r['id']: r for r in read_jsonl(bundle/'source_manifests/original_gear_blur500/source_provenance.jsonl')}
    by_source = {int(r['index']) for r in gear_provenance.values()}
    by_project = {'mc:'+r['project_id'] for r in gear_provenance.values() if r.get('project_id')}
    anchors, projects = {}, {}
    split_priority = {'train': 0, 'val': 1, 'test': 2}
    for catalog in (Path('outputs/qwen_vl_embedding_large_20261103/catalog/samples.jsonl'),
                    Path('data/catalogs/native16_v3/catalog/samples.jsonl')):
        with catalog.open() as f:
            for line in f:
                r = json.loads(line)
                split = r['split']
                if split not in split_priority:
                    continue
                if r.get('source_row') in by_source:
                    index = r['source_row']
                    anchors[index] = max((anchors.get(index, split), split), key=split_priority.get)
                for lineage in r.get('source_lineages', [r['lineage']]):
                    if lineage in by_project:
                        projects[lineage] = max((projects.get(lineage, split), split), key=split_priority.get)
    old = json.loads((legacy/'pairs.json').read_text())
    con = sqlite3.connect('file:outputs/pair_review/review.sqlite3?mode=ro', uri=True)
    con.execute('BEGIN')
    meta = dict(con.execute('select key,value from meta'))
    decisions = {r[0]: {'state': r[1], 'train_selected': bool(r[2])}
                 for r in con.execute('select pair_id,state,train_selected from decisions')}
    con.commit()
    con.close()
    if meta['fingerprint'] != file_hash(legacy/'pairs.json'):
        raise ValueError('Review database does not match original pairs')
    kept = [i for i, r in enumerate(old) if r['split'] == 'train'
            and decisions.get(pair_id(r), {}).get('state') == 'keep']
    if not kept:
        raise ValueError('No manually approved legacy pairs')
    atomic_json(root/'review_snapshot.json', {'revision': int(meta['revision']),
        'source_pairs_sha256': meta['fingerprint'], 'selection_policy': 'review_state_keep',
        'selected_indices': kept, 'selected_pair_ids': [pair_id(old[i]) for i in kept],
        'decisions': {pair_id(old[i]): decisions[pair_id(old[i])] for i in kept}})
    rows, maps = [], {}

    def save_target(array):
        a = native_grid(array, 32)
        digest = content_hash(a)
        path = targets/f'{digest}.png'
        if not path.exists():
            Image.fromarray(a).save(path)
        return a, digest, str(path.resolve())

    # Legacy validation remains evaluation-only; no unreviewed training rows.
    for i in kept+[i for i, r in enumerate(old) if r['split'] == 'val']:
        r = old[i]
        spec = r['source_target']
        if spec['path'] not in maps:
            maps[spec['path']] = np.memmap(spec['path'], mode='r', dtype='uint8', shape=tuple(spec['shape']))
        a, digest, path = save_target(np.array(maps[spec['path']][spec['index']]))
        rows.append({'id': 'reviewed:'+pair_id(r), 'quality_tier': 'reviewed' if i in kept else 'evaluation_only',
            'quality_basis': 'human_keep' if i in kept else 'legacy_validation_only',
            'target_file': path, 'target_hash': digest, 'reference_file': r['reference_file'],
            'reference_sha256': r['reference_sha256'], 'captions': [r['prompt']],
            'split': r['split'], 'split_group': r['split_group'], 'source_row': r['source_row'],
            'source_family': 'reviewed_legacy', 'legacy_feature_index': i,
            'provenance': {'pair_id': pair_id(r), 'source_filename': r.get('source_filename'),
                           'lineage': r['lineage'], 'review_revision': int(meta['revision'])}})
    effective_grids = Counter()
    reference_sizes = Counter()
    for r in new_rows:
        with Image.open(bundle/r['mc_image']) as im:
            a, digest, path = save_target(np.array(im.convert('RGBA')))
        try:
            native_grid(a, 16)
            effective_grids['16x16_lossless'] += 1
        except ValueError:
            effective_grids['32x32_detail'] += 1
        reference_info = image_map[r['hd_image']]
        reference_sizes[f'{reference_info["width"]}x{reference_info["height"]}'] += 1
        p = (gear_provenance if r['batch'] == 'original_gear_blur500' else main_provenance).get(r['original_id'], r['provenance'])
        original_index = p.get('index')
        source_split = anchors.get(original_index) or projects.get('mc:'+p.get('project_id', ''))
        group = ('original:'+p.get('dataset', '')+':'+str(original_index) if original_index is not None
                 else 'design:'+(p.get('source_id') or r['sample_id']))
        # Old holdouts retain their status. New designs receive a stable holdout.
        split = source_split if source_split in ('val', 'test') else ('val' if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)%10 == 0 else 'train')
        rows.append({'id': r['sample_id'], 'quality_tier': 'highest_quality',
            'quality_basis': 'user_curated_primary_1000', 'source_family': r['batch'],
            'target_file': path, 'target_hash': digest,
            'reference_file': str((bundle/r['hd_image']).resolve()),
            'reference_sha256': image_map[r['hd_image']]['sha256'],
            'captions': target_captions(r), 'split': split, 'split_group': group,
            'source_row': original_index, 'previous_split': source_split,
            'mc_annotation_basis': r['mc_annotation_basis'], 'mc_annotation_en': r['mc_annotation_en'],
            'hd_annotation_en': r['hd_annotation_en'], 'name_en': r['name_en'], 'name_zh': r['name_zh'],
            'expanded_generation_prompt': r['expanded_generation_prompt'], 'provenance': p,
            'raw_sample': r})
    # Connected duplicate targets, references and original design identities share
    # one split. Holdouts win over training when any source already held it out.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(rows):
        keys = [('target', r['target_hash']), ('reference', r['reference_sha256'])]
        if r.get('source_row') is not None:
            keys.append(('original', r['source_row']))
        if r['quality_tier'] == 'highest_quality':
            keys.append(('design', r['split_group']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    grouped = defaultdict(list)
    for i in range(len(rows)):
        grouped[find(i)].append(i)
    moves = []
    priority = {'train': 0, 'val': 1, 'test': 2}
    for ids in grouped.values():
        split = max((rows[i]['split'] for i in ids), key=priority.get)
        for i in ids:
            if rows[i]['split'] != split:
                moves.append({'id': rows[i]['id'], 'from': rows[i]['split'], 'to': split})
                rows[i]['split'] = split
    for i, r in enumerate(rows):
        r['feature_index'] = i
    atomic_json(root/'pairs.json', rows)
    counts = Counter((r['quality_tier'], r['split']) for r in rows)
    stats = {'quality_tier': 'highest_quality', 'primary_pairs': 1000, 'excluded_history_or_trials': 93,
        'verified_images': len(verified), 'native_target_size': 32, 'target_resampling': False,
        'review_revision': int(meta['revision']), 'reviewed_selected_pairs': len(kept),
        'counts': {f'{q}/{s}': n for (q, s), n in counts.items()},
        'unique_targets': {s: len({r['target_hash'] for r in rows if r['split'] == s}) for s in ('train', 'val', 'test')},
        'split_adjustments': moves, 'source_manifest_sha256': file_hash(bundle/'metadata/pairs_main_1000.jsonl'),
        'pairs_sha256': file_hash(root/'pairs.json'), 'review_snapshot_sha256': file_hash(root/'review_snapshot.json'),
        'raw_bundle': str(bundle.resolve()), 'archive': str(Path('data/raw/archives/MC_Gen_AllData_20261106.zip').resolve()),
        'caption_policy': 'MC observations/verified MC semantics plus short names; HD observations and generation intent stored separately',
        'verification_seconds': time.monotonic()-start}
    atomic_json(processed/'dataset.json', {'dataset_id': 'MC_Gen_20261106_MAIN_1000',
        'quality_tier': 'highest_quality', 'primary_pairs': 1000,
        'sources': {'mcgen2_items_500': 500, 'original_gear_blur500': 500},
        'splits': dict(Counter(r['split'] for r in rows if r['quality_tier'] == 'highest_quality')),
        'raw_bundle': str(bundle.resolve()), 'primary_manifest': str((bundle/'metadata/pairs_main_1000.jsonl').resolve()),
        'mc_file_canvas': [32, 32], 'mc_effective_grid_counts': dict(effective_grids),
        'hd_reference_file_sizes': dict(reference_sizes), 'vision_encoder_input_canvas': [256, 256],
        'shared_native32_cache': str(targets.resolve()), 'mixed_run_audit': str((root/'data_audit.json').resolve()),
        'caption_policy': stats['caption_policy'], 'verified_archive_images': len(verified)})
    atomic_json(root/'data_audit.json', stats)
    cfg = Experiment.load(legacy/'config.yaml')
    cfg.model.size = cfg.model.ref_size = 32
    cfg.train.trainable = 'joint_finetune'
    cfg.train.steps = 3000
    cfg.train.batch_size = 32
    cfg.train.warmup = 100
    cfg.train.lr = 1e-5
    cfg.train.task_mix = {'text': .18, 'reference': .10, 'joint': .70, 'unconditional': .02}
    cfg.train.val_samples = 24
    cfg.train.precision = 'bfloat16'
    cfg.train.grad_clip = 1.
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    atomic_json(root/'initialization.json', {'checkpoint': str(Path('outputs/approved_reference_trial_20261104/training/best_joint_flow.pt').resolve()),
        'sha256': file_hash('outputs/approved_reference_trial_20261104/training/best_joint_flow.pt'),
        'transfer': 'native16 to native32; bilinear interpolate positional buffer; all learned weights copied exactly',
        'generator_lr': 1e-5, 'visual_adapter_lr': 1e-4, 'frozen_encoders': True,
        'reference_hq_fraction': .6, 'pure_text_source': 'highest_quality only', 'planned_draws_per_target': 120})
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=Path('data/raw/MC_Gen_AllData_20261106'))
    p.add_argument('--legacy', type=Path, default=Path('outputs/spatial_reference_20261104'))
    a = p.parse_args()
    prepare(a.out, a.bundle, a.legacy)
