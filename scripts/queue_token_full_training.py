"""Two-card exact token caching -> real checks -> full-data training -> evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mcgen.catalog import atomic_json,file_hash
from prepare_token_full_training import DEFAULT


def read(path):return json.loads(path.read_text())


def run(root):
    root=root.resolve()
    lock=(root/'token_queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=read(root/'code_verification.json')
    def verify_code():
        if not proof['passed']:raise ValueError('Code verification required')
        for path,digest in proof['files'].items():
            if file_hash(path)!=digest:raise ValueError('Code changed after verification: '+path)
    verify_code()
    workers=[];active=[None];stopping=[False]
    def stop(*_):
        stopping[0]=True
        for child in workers+[active[0]]:
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1',
         'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    atomic_json(root/'token_queue_worker.json',{'pid':os.getpid(),'started_unix':time.time()})
    phase=['starting']
    def status(name,**values):
        phase[0]=name;atomic_json(root/'token_queue_status.json',{'phase':name,'updated_unix':time.time(),**values})
    def stage(name,args):
        if stopping[0]:raise InterruptedError('Queue stopped')
        verify_code();status(name)
        with (root/(name+'.log')).open('a') as log:
            active[0]=subprocess.Popen([sys.executable,'-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
            code=active[0].wait();active[0]=None
        if code:raise RuntimeError(name+' failed; inspect '+name+'.log')
    try:
        plan=read(root/'text_features/plan.json')
        if not read(root/'text_features/metadata.json')['complete']:
            for device in ('cuda:0','cuda:1'):
                log=(root/('cache_'+device.replace(':','')+'.log')).open('a')
                child=subprocess.Popen([sys.executable,'-u','scripts/cache_token_text.py','worker',
                    '--out',str(root),'--device',device],env=env,stdout=log,stderr=subprocess.STDOUT)
                log.close();workers.append(child)
            def progress():
                chunks=[read(p) for p in (root/'text_features/chunks').glob('*.json')]
                done=sum(v['hi']-v['lo'] for v in chunks)
                speeds=[read(p)['recent_prompts_per_second'] for p in (root/'text_features').glob('progress_cuda*.json')]
                status('cache_tokens',done=done,total=plan['count'],probe_passed=(root/'token_condition_check.json').exists(),
                    estimated_remaining_seconds=(plan['count']-done)/sum(speeds) if sum(speeds)>0 else None)
                for child in workers:
                    if child.poll() not in (None,0):raise RuntimeError('Token cache worker failed')
                if stopping[0]:raise InterruptedError('Queue stopped')
            needed=range(0,plan['probe_prompt_count'],plan['chunk_prompts'])
            while not all((root/'text_features/chunks'/f'{i:07d}.json').exists() for i in needed):
                progress();time.sleep(5)
            if not (root/'probe_ready.json').exists():stage('prepare_probe',['scripts/cache_token_text.py','probe','--out',str(root)])
            if not (root/'token_condition_check.json').exists():
                stage('token_condition_check',['scripts/check_token_conditioning.py','--out',str(root),'--device','cuda:1'])
            if not read(root/'token_condition_check.json')['passed']:raise ValueError('Real token learning check failed')
            while any(child.poll() is None for child in workers):progress();time.sleep(10)
            progress()
            stage('finalize_cache',['scripts/cache_token_text.py','finalize','--out',str(root)])
        if not (root/'checks.json').exists() or not read(root/'checks.json')['passed']:
            stage('batch_and_gradient_checks',['scripts/check_five_task_reference.py','--out',str(root),
                '--device','cuda:1','--microbatches','80','160','320','640'])
        stage('training_and_evaluation',['scripts/queue_five_task_reference.py','--out',str(root),
            '--evaluation',str(Path('outputs/token_full_hq_evaluation_20261109').resolve()),'--device','cuda:1'])
        final=read(root/'status.json')
        status('complete' if final.get('complete') else 'stopped',training_step=final.get('step'),
            stop_reason=final.get('stop_reason'),evaluation='outputs/token_full_hq_evaluation_20261109')
    except BaseException as error:
        failed_phase=phase[0];stop()
        status('stopped' if isinstance(error,InterruptedError) else 'failed',failed_phase=failed_phase,error=str(error))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    run(p.parse_args().out)
