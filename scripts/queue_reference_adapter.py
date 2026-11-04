"""Wait for canonical cache, run a bounded real-data check, probe batch, train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mcgen.catalog import atomic_json


def main(root,device):
    status=root/'queue_status.json'
    def phase(name,**extra):atomic_json(status,{'phase':name,'updated_unix':time.time(),**extra})
    def check_cache_worker():
        worker=root/'cache_worker.json'
        if worker.exists() and not Path(f"/proc/{json.load(open(worker))['pid']}").exists() and not (root/'features/metadata.json').exists():
            raise RuntimeError('cache worker exited before completion')
    phase('waiting_for_first_cache_chunk')
    while not (root/'features/000000.pt').exists():
        if (root/'cache_failed.json').exists():raise RuntimeError('embedding cache failed')
        check_cache_worker()
        time.sleep(10)
    # This small check can overlap encoding; the memory-heavy batch probe waits
    # until the encoder process exits, so its batch budget is measured fairly.
    for stage in ('smoke','controls','probe','train'):
        if stage=='controls':
            phase('waiting_for_full_cache')
            while not (root/'features/metadata.json').exists():
                if (root/'cache_failed.json').exists():raise RuntimeError('embedding cache failed')
                check_cache_worker()
                time.sleep(10)
            meta=json.load(open(root/'features/metadata.json'))
            if not meta['complete']:raise RuntimeError('incomplete feature cache')
            # CUDA allocations are owned by the cache worker until process exit.
            pid=json.load(open(root/'cache_worker.json'))['pid']
            while Path(f'/proc/{pid}').exists():time.sleep(2)
        phase(stage)
        with (root/f'{stage}.log').open('w') as f:
            script='scripts/cache_reference_adapter.py' if stage=='controls' else 'scripts/train_reference_adapter.py'
            code=subprocess.call([sys.executable,'-u',script,stage,'--out',str(root),'--device',device],stdout=f,stderr=subprocess.STDOUT)
        if code:phase('failed',stage=stage,returncode=code);raise SystemExit(code)
    phase('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    try:main(a.out,a.device)
    except Exception as exc:
        atomic_json(a.out/'queue_status.json',{'phase':'failed','error':str(exc)});raise
