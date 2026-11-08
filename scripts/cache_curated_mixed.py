"""Resumable exact visual/text feature caches for the reviewed + HQ experiment."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.tasks import conditioned_prompts
from mcgen.visual_tokens import VisualTokenEncoder


def cache(root, device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.38, device)
    cfg = Experiment.load(root/'config.yaml')
    rows = json.loads((root/'pairs.json').read_text())
    pair_hash = file_hash(root/'pairs.json')
    old_root = Path('outputs/spatial_reference_20261104/visual_features')
    old_meta = json.loads((old_root/'metadata.json').read_text())
    vision = VisualTokenEncoder(cfg.text, device)
    if vision.identity != old_meta['visual_identity'] or not old_meta['complete']:
        raise ValueError('Legacy visual cache uses a different encoder protocol')
    old = np.memmap(old_root/'features.uint16.mmap', mode='r', dtype='uint16', shape=tuple(old_meta['shape']))
    path = root/'visual_features'
    path.mkdir(exist_ok=True)
    shape = [len(rows), cfg.model.visual_grid**2, cfg.model.visual_dim]
    meta = {'complete': False, 'shape': shape, 'visual_identity': vision.identity,
            'pairs_sha256': pair_hash, 'chunks': {}}
    if (path/'metadata.json').exists():
        saved = json.loads((path/'metadata.json').read_text())
        for key in ('shape', 'visual_identity', 'pairs_sha256'):
            if saved[key] != meta[key]:
                raise ValueError('Changed visual cache contract')
        meta = saved
    output = path/'features.uint16.mmap'
    mmap = np.memmap(output, mode='r+' if output.exists() else 'w+', dtype='uint16', shape=tuple(shape))
    start = time.monotonic()
    verified_old_chunks = set()
    for offset in range(0, len(rows), 32):
        stop = min(offset+32, len(rows))
        key = str(offset)
        if key in meta['chunks']:
            if hashlib.sha256(np.asarray(mmap[offset:stop]).tobytes()).hexdigest() != meta['chunks'][key]:
                raise ValueError('Visual cache changed')
            continue
        block = torch.zeros(stop-offset, shape[1], shape[2], dtype=torch.bfloat16)
        fresh_images, positions = [], []
        for local, row in enumerate(rows[offset:stop]):
            if 'legacy_feature_index' in row:
                i = row['legacy_feature_index']
                chunk = i//64*64
                if chunk not in verified_old_chunks:
                    digest = hashlib.sha256(np.asarray(old[chunk:chunk+64]).tobytes()).hexdigest()
                    if digest != old_meta['chunks'][str(chunk)]:
                        raise ValueError('Legacy spatial cache changed')
                    verified_old_chunks.add(chunk)
                block[local] = torch.from_numpy(np.array(old[i])).view(torch.bfloat16)
            else:
                if file_hash(row['reference_file']) != row['reference_sha256']:
                    raise ValueError('New reference changed')
                with Image.open(row['reference_file']) as im:
                    fresh_images.append(im.convert('RGBA'))
                positions.append(local)
        if fresh_images:
            block[positions] = vision.encode(fresh_images)
        array = block.view(torch.uint16).numpy()
        mmap[offset:stop] = array
        mmap.flush()
        meta['chunks'][key] = hashlib.sha256(array.tobytes()).hexdigest()
        atomic_json(path/'metadata.json', meta)
        atomic_json(root/'cache_progress.json', {'phase': 'visual', 'done': stop,
                    'total': len(rows), 'seconds': time.monotonic()-start})
        print('VISUAL', stop, len(rows), flush=True)
    meta['complete'] = True
    atomic_json(path/'metadata.json', meta)
    del vision, old, mmap
    gc.collect()
    torch.cuda.empty_cache()
    prompts = list(dict.fromkeys(conditioned_prompts(
        [p for row in rows for p in row['captions']]+['helmet', 'sword', 'shield', 'axe'], cfg)))
    text_root = root/'text_features'
    text_root.mkdir(exist_ok=True)
    atomic_json(text_root/'prompts.json', prompts)
    old_text = DiskTextEncoder('outputs/qwen_vl_embedding_large_20261103/features')
    encoder = TextEncoder(cfg.text, cfg.model, device)
    if encoder.identity != old_text.identity:
        raise ValueError('Text encoder identity changed')
    text_shape = [len(prompts), 1, cfg.model.text_dim]
    meta = {'complete': False, 'shape': text_shape, 'text_identity': encoder.identity,
            'prompts_sha256': file_hash(text_root/'prompts.json'), 'done': 0,
            'new_caption_batch_policy': 'singleton; identical to interactive single-prompt inference'}
    if (text_root/'metadata.json').exists():
        saved = json.loads((text_root/'metadata.json').read_text())
        for key in ('shape', 'text_identity', 'prompts_sha256'):
            if saved[key] != meta[key]:
                raise ValueError('Changed text cache contract')
        meta = saved
    feature_file = text_root/'features.uint16.mmap'
    mask_file = text_root/'masks.uint8.mmap'
    values = np.memmap(feature_file, mode='r+' if feature_file.exists() else 'w+', dtype='uint16', shape=tuple(text_shape))
    masks = np.memmap(mask_file, mode='r+' if mask_file.exists() else 'w+', dtype='uint8', shape=tuple(text_shape[:2]))
    for offset in range(meta['done'], len(prompts)):
        prompt = prompts[offset]
        h, m = (old_text if prompt in old_text.index else encoder).encode([prompt], 'cpu')
        values[offset] = h[0].contiguous().view(torch.uint16).numpy()
        masks[offset] = m[0].numpy().astype('uint8')
        if offset%32 == 31 or offset == len(prompts)-1:
            values.flush()
            masks.flush()
            meta['done'] = offset+1
            atomic_json(text_root/'metadata.json', meta)
            atomic_json(root/'cache_progress.json', {'phase': 'text', 'done': offset+1,
                        'total': len(prompts), 'seconds': time.monotonic()-start})
            print('TEXT', offset+1, len(prompts), flush=True)
    meta['complete'] = True
    meta['features_sha256'] = file_hash(feature_file)
    meta['masks_sha256'] = file_hash(mask_file)
    atomic_json(text_root/'metadata.json', meta)
    atomic_json(root/'cache_progress.json', {'phase': 'complete', 'done': len(prompts),
                'total': len(prompts), 'seconds': time.monotonic()-start})
    print('CACHE COMPLETE', round(time.monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cache(a.out, a.device)
