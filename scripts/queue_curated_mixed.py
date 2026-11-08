"""Detached queue for verified HQ mixed reference/text training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mcgen.catalog import atomic_json


def main(root, device, memory_fraction=.38):
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root/'queue_worker.json', {'pid': os.getpid(), 'device': device,
                                             'memory_fraction': memory_fraction})

        def phase(name, **extra):
            atomic_json(root/'queue_status.json', {'phase': name, 'updated_unix': time.time(), **extra})

        phase('waiting_for_cache')
        while True:
            paths = [root/'visual_features/metadata.json', root/'text_features/metadata.json']
            if all(p.exists() and json.loads(p.read_text())['complete'] for p in paths):
                break
            worker = root/'cache_worker.json'
            if worker.exists() and not Path(f"/proc/{json.loads(worker.read_text())['pid']}").exists():
                raise RuntimeError('Cache worker exited before completion')
            time.sleep(5)
        phase('waiting_for_code_checks')
        while not (root/'verification.json').exists():
            time.sleep(5)
        if not json.loads((root/'verification.json').read_text())['passed']:
            raise RuntimeError('Code verification failed')
        for stage, artifact in [('smoke', 'smoke.json'), ('probe', 'batch_budget.json'), ('train', None)]:
            if artifact and (root/artifact).exists():
                if stage == 'smoke' and not json.loads((root/artifact).read_text())['passed']:
                    raise RuntimeError('Real-data mixed learning check failed')
                continue
            phase(stage)
            with (root/f'{stage}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/train_curated_mixed.py', stage,
                    '--out', str(root), '--device', device, '--memory-fraction', str(memory_fraction)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        phase('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.38)
    a = p.parse_args()
    try:
        main(a.out, a.device, a.memory_fraction)
    except Exception as exc:
        atomic_json(a.out/'queue_status.json', {'phase': 'failed', 'error': str(exc)})
        raise
