"""Run verified recovery checks and training in a resumable detached queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mcgen.catalog import atomic_json


def main(root,device):
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(root/'queue_worker.json',{'pid':os.getpid(),'device':device})
        if not json.loads((root/'preparation.json').read_text())['complete']:
            raise ValueError('Preparation incomplete')
        if not json.loads((root/'code_verification.json').read_text())['passed']:
            raise ValueError('Code checks failed')
        for stage in ('check','train'):
            if stage=='check' and (root/'checks.json').exists():
                if not json.loads((root/'checks.json').read_text())['passed']:
                    raise ValueError('Real-data training checks previously failed')
                continue
            atomic_json(root/'queue_status.json',{'phase':stage,'updated_unix':time.time()})
            with (root/f'{stage}.log').open('a') as log:
                subprocess.run([sys.executable,'-u','scripts/train_reference_recovery.py',stage,
                    '--out',str(root),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
        status=json.loads((root/'status.json').read_text())
        atomic_json(root/'queue_status.json',{'phase':'complete' if status['complete'] else 'stopped',
                                            'updated_unix':time.time(),'step':status['step']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_recovery_20261108'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    try:main(args.out,args.device)
    except Exception as exc:
        atomic_json(args.out/'queue_status.json',{'phase':'failed','error':str(exc)})
        raise
