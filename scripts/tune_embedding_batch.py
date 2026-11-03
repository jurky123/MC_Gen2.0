"""Select stable batch by measured memory and throughput, then derive exposure budget."""
import json,math,os,subprocess,sys
from pathlib import Path
import yaml
from mcgen.catalog import atomic_json
from mcgen.config import Experiment
from mcgen.model import PixelFlow
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')
if __name__=='__main__':
 candidates=[64,128,256,384,512,768,1024,1280];rows=[]
 def probe(b):
  with (ROOT/'batch_probe.log').open('a') as log:subprocess.run([sys.executable,'-u','scripts/probe_embedding_batch.py','--batch',str(b)],stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4'})
  r=json.load(open(ROOT/'batch_probe'/f'batch_{b}.json'));rows.append(r);print(r,flush=True);return r
 last_ok=None
 for b in candidates:
  r=probe(b)
  if r['status']=='oom':
   if last_ok is not None:
    midpoint=((last_ok+b)//2//64)*64
    if midpoint>last_ok:probe(midpoint)
   break
  last_ok=b
 valid=[r for r in rows if r['status']=='ok' and max(r['train_peak_gib'],r['eval_peak_gib'])<=70]
 assert valid,'No stable batch found'
 fastest=max(r['samples_per_second'] for r in valid);chosen=max((r for r in valid if r['samples_per_second']>=.95*fastest),key=lambda r:r['batch_size']);batch=chosen['batch_size'];n=311510;prob=.9;steps=math.ceil(14*n/(batch*prob))
 cfg=yaml.safe_load((ROOT/'config.yaml').read_text());cfg['train'].update(batch_size=batch,accumulation=1,steps=steps,warmup=math.ceil(.5*n/batch),eval_every=math.ceil(n/(prob*batch)))
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));Experiment.load(ROOT/'config.yaml')
 report={'selected':chosen,'all_trials':rows,'selection':'largest stable batch within5% of best measured throughput; <=70GiB allocated peak, process allocator cap90%','train_targets':n,'effective_batch':batch,'steps':steps,'expected_total_exposures':steps*batch/n,'expected_text_exposures':steps*batch*prob/n,'warmup_updates':cfg['train']['warmup'],'eval_every_updates':cfg['train']['eval_every'],'encoder':'Qwen3-VL-Embedding-8B official pooled vector','text_tokens':1}
 atomic_json(ROOT/'batch_and_budget.json',report);print('SELECTED',json.dumps(report),flush=True)
