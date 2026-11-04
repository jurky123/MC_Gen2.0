"""Resume-safe spatial adapter queue: complete cache, learning check, batch, train."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mcgen.catalog import atomic_json


def main(root, device):
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root/'queue_worker.json', {'pid': os.getpid(), 'device': device})

        def phase(name, **extra):
            atomic_json(root/'queue_status.json', {'phase': name, 'updated_unix': time.time(), **extra})

        phase('waiting_for_spatial_cache')
        while True:
            path = root/'visual_features/metadata.json'
            if path.exists() and json.loads(path.read_text())['complete']:
                break
            worker = root/'cache_worker.json'
            if worker.exists() and not Path(f"/proc/{json.loads(worker.read_text())['pid']}").exists():
                raise RuntimeError('Spatial cache worker exited before completion')
            time.sleep(5)
        # Local release checks may be recorded here; fresh runs still execute
        # the mandatory real-data smoke below without a hand-written marker.
        checks_path = root/'verification.json'
        if checks_path.exists() and not json.loads(checks_path.read_text())['passed']:
            raise RuntimeError('Recorded code checks did not pass')
        for stage, artifact in [('smoke', 'smoke.json'), ('probe', 'batch_budget.json'), ('train', None)]:
            if artifact and (root/artifact).exists():
                if stage == 'smoke' and not json.loads((root/artifact).read_text())['passed']:
                    raise RuntimeError('Previous real-data learning check failed')
                continue
            phase(stage)
            with (root/f'{stage}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/train_visual_reference.py', stage,
                    '--out', str(root), '--device', device], stdout=log, stderr=subprocess.STDOUT, check=True)
        phase('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    try:
        main(a.out, a.device)
    except Exception as exc:
        atomic_json(a.out/'queue_status.json', {'phase': 'failed', 'error': str(exc)})
        raise
