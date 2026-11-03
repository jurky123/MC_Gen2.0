"""Wait for pinned annotator files, verify them, then run a bounded annotation batch."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mcgen.catalog import atomic_json, file_hash

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    status=out/'run_status.json'
    def update(stage,**extra):
        record={'stage':stage,'updated_unix':time.time(),**extra};atomic_json(status,record)
        print(json.dumps(record),flush=True)
    def command(stage,cmd):
        update(stage)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'4'}
        with (out/f'{stage}.log').open('a') as log:
            subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        last=None
        while True:
            if not args.manifest.exists():
                pending=['manifest.json']
            else:
                manifest=json.loads(args.manifest.read_text())
                pending=[f['name'] for f in manifest['files'] if not (args.model/f['name']).is_file()
                         or (args.model/f['name']).stat().st_size!=f['size']]
            if not pending:break
            if pending!=last:update('waiting_for_download',pending=pending);last=pending
            time.sleep(15)
        update('verifying_download')
        checked=[]
        for f in manifest['files']:
            digest=file_hash(args.model/f['name'])
            if f['sha256'] and digest!=f['sha256']:raise ValueError(f"Hash mismatch: {f['name']}")
            checked.append({**f,'verified_sha256':digest})
        atomic_json(out/'download/verified.json',dict(manifest,files=checked))
        common=['scripts/audit_item_captions.py','annotate','--out',str(out),'--device',args.device]
        command('vision_smoke',common+['--limit','2','--batch-size','2'])
        candidates=[json.loads(x) for x in (out/'candidates.jsonl').read_text().splitlines()]
        if any(r.get('annotation') is None for r in candidates[:2]):
            raise ValueError('Smoke output failed schema; inspect candidates before full batch')
        command('vision_batch',common+['--batch-size','16'])
        command('visual_review',['scripts/audit_item_captions.py','review','--out',str(out)])
        candidates=[json.loads(x) for x in (out/'candidates.jsonl').read_text().splitlines()]
        cases=[json.loads(x) for x in (out/'cases.jsonl').read_text().splitlines()]
        if {r['id'] for r in candidates}!={r['id'] for r in cases}:raise ValueError('Incomplete vision batch')
        command('semantic_batch',['scripts/name_item_captions.py','--input',str(out/'candidates.jsonl'),
                                  '--out',str(out/'semantic'),'--device',args.device])
        merged=[json.loads(x) for x in (out/'semantic/candidates.jsonl').read_text().splitlines()]
        if {r['id'] for r in merged}!={r['id'] for r in cases}:raise ValueError('Incomplete semantic batch')
        update('annotation_complete_pending_review',cases=len(cases),
               visual_statuses=dict(Counter(r['status'] for r in candidates)),
               semantic_statuses=dict(Counter(r['status'] for r in merged)),approved_for_training=0)
    except Exception as exc:
        update('failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/iflab/models/Qwen3-VL-8B-Instruct'))
    p.add_argument('--device',default='cuda:1');run(p.parse_args())
