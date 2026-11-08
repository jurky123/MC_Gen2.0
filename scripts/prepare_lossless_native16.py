"""Create a new native16 run, excluding every target that needs lossy resizing."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.curated_training import MixedData, mixed_training_budget
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import content_hash, native_grid, rgba
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists():
        raise ValueError('Use a new run directory; existing experiments are never overwritten')
    originals = json.loads((source/'pairs.json').read_text())
    source_digest = file_hash(source/'pairs.json')
    source_audit = json.loads((source/'data_audit.json').read_text())
    if source_audit['pairs_sha256'] != source_digest:
        raise ValueError('Source pairs changed')
    if source_audit['review_snapshot_sha256'] != file_hash(source/'review_snapshot.json'):
        raise ValueError('Source review snapshot changed')
    info = json.loads((source/'initialization.json').read_text())
    if file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Original16 initialization checkpoint changed')
    state = load_checkpoint(info['checkpoint'])
    if state['config']['model']['size'] != 16:
        raise ValueError('Restart must use the original16 checkpoint, never the32-trained model')
    del state
    out.mkdir(parents=True)
    target_root = Path('data/processed/reference_native16_lossless_20261107/targets').resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    restored, excluded, rows, source_indices = {}, [], [], []
    for index, original in enumerate(originals):
        key = original['target_hash']
        if key not in restored:
            with Image.open(original['target_file']) as im:
                pixels = rgba(im)
            if pixels.shape != (32,32,4) or content_hash(pixels) != key:
                raise ValueError('Source target changed')
            try:
                small = native_grid(pixels, 16)
            except ValueError:
                restored[key] = None
            else:
                if not np.array_equal(pixels, small.repeat(2,0).repeat(2,1)):
                    raise ValueError('Lossless roundtrip failed')
                digest = content_hash(small)
                path = target_root/f'{digest}.png'
                if path.exists():
                    with Image.open(path) as im:
                        if content_hash(im) != digest:
                            raise ValueError('Existing restored target changed')
                else:
                    Image.fromarray(small).save(path)
                restored[key] = (digest, str(path))
        if restored[key] is None:
            excluded.append({'source_pair_index':index, 'id':original['id'],
                'source_family':original['source_family'], 'split':original['split'],
                'target_hash32':key, 'reason':'genuine32_grid_would_require_lossy_conversion'})
            continue
        digest, path = restored[key]
        row = copy.deepcopy(original)
        row.update(target_file=path, target_hash=digest, feature_index=len(rows),
            source_pair_index=index, source_feature_index=index,
            source_target_file32=original['target_file'], source_target_hash32=key,
            target_conversion={'method':'exact_repeat_collapse','lossless':True,
                               'source_canvas':[32,32],'output_canvas':[16,16]})
        rows.append(row)
        source_indices.append(index)
    owners = {}
    for row in rows:
        keys = [('target',row['target_hash']), ('reference',row['reference_sha256'])]
        if row.get('source_row') is not None:
            keys.append(('original',row['source_row']))
        for key in keys:
            if owners.setdefault(key,row['split']) != row['split']:
                raise ValueError('Lossless subset has cross-split leakage')
    atomic_json(out/'pairs.json', rows)
    atomic_json(out/'excluded_pairs.json', excluded)
    shutil.copy2(source/'review_snapshot.json', out/'review_snapshot.json')
    cfg = Experiment.load(source/'config.yaml')
    cfg.model.size = cfg.model.ref_size = 16
    cfg.train.batch_size = 128
    train_targets = len({r['target_hash'] for r in rows if r['split']=='train'})
    cfg.train.steps = mixed_training_budget(train_targets, cfg.train.batch_size, 120)
    cfg.train.warmup = min(100, max(1,cfg.train.steps//10))
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
    info.update(transfer='native16 to native16; every learned weight and position buffer copied exactly',
                source_run=str(source), source_pairs_sha256=source_digest,
                planned_draws_per_target=120, excluded_genuine32_targets=True,
                resume_native32_training=False)
    atomic_json(out/'initialization.json', info)
    pair_digest = file_hash(out/'pairs.json')
    visual_root = source/'visual_features'
    visual_meta = json.loads((visual_root/'metadata.json').read_text())
    if not visual_meta['complete'] or visual_meta['pairs_sha256'] != source_digest:
        raise ValueError('Source visual cache contract changed')
    if visual_meta['shape'][0] != len(originals):
        raise ValueError('Source feature count mismatch')
    old = np.memmap(visual_root/'features.uint16.mmap',mode='r',dtype='uint16',shape=tuple(visual_meta['shape']))
    if set(visual_meta['chunks']) != {str(i) for i in range(0,len(originals),32)}:
        raise ValueError('Source cache chunks incomplete')
    for offset, digest in visual_meta['chunks'].items():
        start = int(offset)
        if hashlib.sha256(np.asarray(old[start:start+32]).tobytes()).hexdigest() != digest:
            raise ValueError('Source visual features changed')
    visual_dir = out/'visual_features'
    visual_dir.mkdir()
    shape = [len(rows),*visual_meta['shape'][1:]]
    values = np.memmap(visual_dir/'features.uint16.mmap',mode='w+',dtype='uint16',shape=tuple(shape))
    chunks = {}
    for offset in range(0,len(rows),32):
        block = np.asarray(old[source_indices[offset:offset+32]])
        values[offset:offset+32] = block
        chunks[str(offset)] = hashlib.sha256(block.tobytes()).hexdigest()
    values.flush()
    atomic_json(visual_dir/'metadata.json',{'complete':True,'shape':shape,
        'visual_identity':visual_meta['visual_identity'],'pairs_sha256':pair_digest,'chunks':chunks,
        'source_cache':str(visual_root), 'source_pair_indices':source_indices,
        'protocol':'exact_subset_no_encoder_or_image_transform_change'})
    del old, values
    old_text = DiskTextEncoder(source/'text_features')
    old_text_meta = json.loads((source/'text_features/metadata.json').read_text())
    for name,key in [('features.uint16.mmap','features_sha256'),('masks.uint8.mmap','masks_sha256')]:
        if file_hash(source/'text_features'/name) != old_text_meta[key]:
            raise ValueError('Source text features changed')
    prompts = list(dict.fromkeys(conditioned_prompts(
        [label for row in rows for label in row['captions']]+['helmet','sword','shield','axe'],cfg)))
    text_dir = out/'text_features'
    text_dir.mkdir()
    atomic_json(text_dir/'prompts.json',prompts)
    indices = [old_text.index[p] for p in prompts]
    text_shape = [len(prompts),*old_text.features.shape[1:]]
    features = np.memmap(text_dir/'features.uint16.mmap',mode='w+',dtype='uint16',shape=tuple(text_shape))
    masks = np.memmap(text_dir/'masks.uint8.mmap',mode='w+',dtype='uint8',shape=tuple(text_shape[:2]))
    features[:] = old_text.features[indices]
    masks[:] = old_text.masks[indices]
    features.flush()
    masks.flush()
    atomic_json(text_dir/'metadata.json',{'complete':True,'shape':text_shape,'done':len(prompts),
        'text_identity':old_text.identity, 'prompts_sha256':file_hash(text_dir/'prompts.json'),
        'features_sha256':file_hash(text_dir/'features.uint16.mmap'),
        'masks_sha256':file_hash(text_dir/'masks.uint8.mmap'),
        'source_cache':str(source/'text_features'),'protocol':'exact_prompt_subset'})
    counts = Counter('/'.join((r['quality_tier'],r['source_family'],r['split'])) for r in rows)
    audit = {'pairs_sha256':pair_digest,'review_snapshot_sha256':file_hash(out/'review_snapshot.json'),
        'source_run':str(source),'source_pairs_sha256':source_digest,
        'source_pairs':len(originals),'retained_pairs':len(rows),'excluded_pairs':len(excluded),
        'counts':dict(counts),'unique_targets':{s:len({r['target_hash'] for r in rows if r['split']==s}) for s in ('train','val','test')},
        'hq_retained':sum(r['quality_tier']=='highest_quality' for r in rows),
        'native_target_size':16,'lossy_target_count':0,'all_restored_roundtrips_exact':True,
        'cross_split_target_reference_and_original_collisions':0,
        'captions_references_and_source_splits_unchanged':True,
        'visual_features_exact_source_subset':True, 'text_features_exact_source_subset':True,
        'old_data_and_checkpoints_overwritten':False}
    atomic_json(out/'data_audit.json',audit)
    data = MixedData(out)
    if len(data.rows) != len(rows) or data.targets.shape[-2:] != (16,16):
        raise ValueError('Training loader validation failed')
    atomic_json(out/'cache_progress.json',{'phase':'complete','source':'verified_exact_subsets'})
    atomic_json(out/'preparation.json',{'complete':True,**audit})
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/reference_mixed_hq_20261107'))
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_native16_lossless_20261107'))
    args = parser.parse_args()
    prepare(args.source,args.out)
