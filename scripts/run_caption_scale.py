"""Run the accepted two-observation caption workflow; stop for fixed visual review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from mcgen.catalog import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def run(out,device,batch):
    out=out.resolve()
    while not (out/'vision/protocol.json').exists():time.sleep(10)
    def status(stage,**extra):
        payload={'stage':stage,'updated_unix':time.time(),**extra};atomic_json(out/'run_status.json',payload)
        print(json.dumps(payload),flush=True)
    def command(stage,args):
        status(stage)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'4'}
        if stage!='export':
            while torch.cuda.mem_get_info(device)[0]<23*1024**3:
                status('waiting_for_gpu',next_stage=stage);time.sleep(30)
            status(stage)
        with (out/f'{stage}.log').open('a') as f:
            subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        base=['scripts/audit_item_captions.py','annotate','--out',str(out/'vision'),'--device',device,
              '--batch-size',str(batch),'--memory-fraction','0.30']
        command('vision_smoke',base+['--limit',str(batch)])
        command('vision_batch',base)
        command('refined_batch',['scripts/refine_item_captions.py','--source',str(out/'vision'),
            '--out',str(out/'refined'),'--device',device,'--batch-size',str(batch)])
        command('export',['scripts/export_caption_batch.py','--out',str(out/'refined')])
        cases=[json.loads(s) for s in (out/'vision/cases.jsonl').read_text().splitlines()]
        result=[json.loads(s) for s in (out/'refined/semantic/candidates.jsonl').read_text().splitlines()]
        if len(result)!=len(cases) or {r['id'] for r in cases}!={r['id'] for r in result}:raise ValueError('Incomplete batch')
        status('annotation_complete_pending_fixed_review',records=len(result),generation_training_started=False)
    except Exception as exc:
        status('failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=16);a=p.parse_args();torch.set_num_threads(2);run(a.out,a.device,a.batch_size)
