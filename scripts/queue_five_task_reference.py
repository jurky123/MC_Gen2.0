"""Run balanced training then automatically publish a full held-out evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mcgen.catalog import atomic_json, file_hash
from prepare_five_task_reference import DEFAULT


def main(root, evaluation, device):
    root = root.resolve()
    gallery = Path('outputs/native16_data_review_20261107')
    publish = gallery/('five-task-broad-replay-evaluation' if (root/'text_replay.json').exists() else 'five-task-evaluation')
    recipe=json.loads((root/'recipe.json').read_text())
    if recipe.get('evaluation_gallery_name'):publish=gallery/recipe['evaluation_gallery_name']
    handle = (root/'queue.lock').open('w')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = json.loads((root/'code_verification.json').read_text())
    if not proof['passed']:
        raise ValueError('Code verification must pass')
    for name, digest in proof['files'].items():
        if file_hash(name) != digest:
            raise ValueError('Code changed after verification: '+name)
    if not json.loads((root/'checks.json').read_text())['passed']:
        raise ValueError('GPU checks must pass')
    active = [None]
    stopping = [False]

    def stop(*_):
        stopping[0] = True
        if active[0] is not None and active[0].poll() is None:
            active[0].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    atomic_json(root/'queue_worker.json', {'pid': os.getpid(), 'device': device, 'started_unix': time.time()})

    def stage(name, args):
        if stopping[0]:
            raise InterruptedError('Queue stopped')
        atomic_json(root/'queue_status.json', {'phase': name, 'updated_unix': time.time()})
        with (root/(name+'.log')).open('a') as log:
            active[0] = subprocess.Popen([sys.executable, '-u', *args], stdout=log, stderr=subprocess.STDOUT)
            result = active[0].wait()
        active[0] = None
        if result:
            raise RuntimeError(f'{name} exited with code{result}; see {name}.log')

    try:
        status = root/'status.json'
        if not status.exists() or not json.loads(status.read_text()).get('complete'):
            stage('train', ['scripts/train_five_task_reference.py', '--out', str(root), '--device', device])
        if not json.loads(status.read_text()).get('complete'):
            stopped = json.loads(status.read_text())
            atomic_json(root/'queue_status.json', {'phase': 'stopped', 'updated_unix': time.time(),
                'stop_reason': stopped.get('stop_reason'), 'step': stopped.get('step')})
            return
        evaluation = evaluation.resolve()
        base = ['scripts/evaluate_five_task_reference.py']
        common = ['--root', str(root), '--out', str(evaluation)]
        if not (evaluation/'protocol.json').exists():
            stage('evaluate_prepare', base+['prepare']+common)
        for variant in ('previous', 'best', 'last'):
            progress = evaluation/f'progress_{variant}.json'
            if progress.exists() and json.loads(progress.read_text()).get('phase') == 'complete':
                continue
            stage('evaluate_'+variant, base+['generate']+common+['--variant', variant, '--device', device])
        stage('evaluate_render', base+['render']+common+['--publish', str(publish)])
        atomic_json(root/'queue_status.json', {'phase': 'complete', 'updated_unix': time.time(),
            'training_steps': json.loads((root/'recipe.json').read_text())['steps'], 'evaluation': str(evaluation),
            'url': 'http://localhost:8766/'+publish.name+'/'})
    except BaseException as error:
        atomic_json(root/'queue_status.json', {'phase': 'stopped' if stopping[0] else 'failed',
            'error': str(error), 'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--evaluation', type=Path, default=Path('outputs/reference_five_task_evaluation_20261108'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    main(args.out, args.evaluation, args.device)
