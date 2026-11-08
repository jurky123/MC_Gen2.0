"""Coordinate disjoint evaluation shards, preserving the FP32 protocol."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from PIL import Image
from mcgen.catalog import atomic_json


def main(root, pid, device):
    with (root/'detail.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', 'scripts/validate_resolution.py', 'detail',
            '--out', str(root), '--device', device], stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        last = root/'native/common_07_post32_s1.png'
        while True:
            if last.exists():
                try:
                    with Image.open(last) as im:
                        im.verify()
                    break
                except Exception:
                    pass
            if not Path(f'/proc/{pid}').exists():
                raise RuntimeError('Main evaluation worker exited before its common-prompt shard completed')
            time.sleep(.5)
        # The original worker next enters a duplicate detailed-caption loop.
        # Stop only that recorded worker, after all its main outputs are saved.
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'scripts/validate_resolution.py' not in command or b'post32' not in command:
            raise RuntimeError('Recorded evaluation worker identity changed')
        os.kill(pid, signal.SIGTERM)
        if child.wait():
            raise RuntimeError('Detailed-caption shard failed')
    rows = json.loads((root/'samples.json').read_text())
    for row in rows:
        for mode in ('text', 'reference', 'joint'):
            for seed in (0, 1):
                with Image.open(root/'native'/f'{row["case_id"]}_post32_{mode}_s{seed}.png') as im:
                    if im.size != (32, 32):
                        raise ValueError('Invalid native output')
    atomic_json(root/'progress_post32.json', {'phase': 'complete', 'shards': 'main + detailed captions, same FP32 protocol'})
    print('POST32 SHARDS COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--main-pid', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    try:
        main(a.out, a.main_pid, a.device)
    except Exception as exc:
        atomic_json(a.out/'shard_failure.json', {'error': str(exc)})
        raise
