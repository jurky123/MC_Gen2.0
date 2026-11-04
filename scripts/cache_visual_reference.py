"""Cache frozen spatial tokens for the unchanged canonical HD/MC pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.runtime import load_checkpoint
from mcgen.visual_tokens import VisualTokenEncoder, TOKEN_GRID, FEATURE_DIM


def run(out, source, device, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'cache_worker.json', {'pid': os.getpid()})
    pairs_path = source/'pairs.json'
    audit = json.loads((source/'data_audit.json').read_text())
    if file_hash(pairs_path) != audit['pairs_sha256']:
        raise ValueError('Canonical pairs changed')
    if (out/'pairs.json').exists():
        if file_hash(out/'pairs.json') != audit['pairs_sha256']:
            raise ValueError('Output belongs to different pairs')
    else:
        (out/'pairs.json').write_bytes(pairs_path.read_bytes())
    atomic_json(out/'data_audit.json', audit)
    cfg = Experiment.from_dict(load_checkpoint(audit['base_checkpoint'])['config'])
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.12, device)
    encoder = VisualTokenEncoder(cfg.text, device)
    rows = json.loads(pairs_path.read_text())
    root = out/'visual_features'
    root.mkdir(exist_ok=True)
    shape = (len(rows), TOKEN_GRID**2, FEATURE_DIM)
    metadata = {'complete': False, 'shape': list(shape), 'visual_identity': encoder.identity,
                'pairs_sha256': audit['pairs_sha256'], 'dtype': 'bfloat16', 'chunks': {}}
    if (root/'metadata.json').exists():
        old = json.loads((root/'metadata.json').read_text())
        for k in ('shape', 'visual_identity', 'pairs_sha256', 'dtype'):
            if old[k] != metadata[k]:
                raise ValueError('Visual cache contract changed: '+k)
        metadata = old
    path = root/'features.uint16.mmap'
    if path.exists() and path.stat().st_size != int(np.prod(shape))*2:
        raise ValueError('Visual cache size changed')
    mmap = np.memmap(path, mode='r+' if path.exists() else 'w+', dtype='uint16', shape=shape)
    start = time.monotonic()
    processed = 0
    for offset in range(0, len(rows), 64):
        stop = min(len(rows), offset+64)
        key = str(offset)
        if key in metadata['chunks']:
            if hashlib.sha256(np.asarray(mmap[offset:stop]).tobytes()).hexdigest() != metadata['chunks'][key]:
                raise ValueError('Cached spatial features changed')
            continue
        images = []
        for row in rows[offset:stop]:
            if file_hash(row['reference_file']) != row['reference_sha256']:
                raise ValueError('Reference file changed')
            with Image.open(row['reference_file']) as im:
                images.append(im.convert('RGBA'))
        values = encoder.encode(images)
        array = values.contiguous().view(torch.uint16).numpy()
        mmap[offset:stop] = array
        mmap.flush()
        metadata['chunks'][key] = hashlib.sha256(array.tobytes()).hexdigest()
        atomic_json(root/'metadata.json', metadata)
        processed += stop-offset
        elapsed = time.monotonic()-start
        progress = {'done': stop, 'total': len(rows), 'processed_this_run': processed,
                    'seconds': elapsed, 'estimated_remaining_seconds': elapsed/processed*(len(rows)-stop)}
        atomic_json(out/'cache_progress.json', progress)
        print('VISUAL CACHE', stop, len(rows), round(elapsed, 1), flush=True)
        if limit is not None and processed >= limit:
            return
    metadata['complete'] = True
    atomic_json(root/'metadata.json', metadata)
    print('VISUAL CACHE COMPLETE', round(time.monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('outputs/reference_adapter_20261104'))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    run(a.out, a.source, a.device, a.limit)
