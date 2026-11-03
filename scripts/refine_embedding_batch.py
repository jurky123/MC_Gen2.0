"""Refine the measured640/768 memory boundary before feature-cache completion."""
import json,math,os,subprocess,sys
from pathlib import Path
import yaml
from mcgen.catalog import atomic_json
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')
if __name__=='__main__':
 report=json.load(open(ROOT/'batch_and_budget.json'));rows=report['all_trials']
 for b in [672,704]:
  with (ROOT/'batch_probe.log').open('a') as log:subprocess.run([sys.executable,'-u','scripts/probe_embedding_batch.py','--batch',str(b)],stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4'})
  rows.append(json.load(open(ROOT/'batch_probe'/f'batch_{b}.json')))
 valid=[r for r in rows if r['status']=='ok' and max(r['train_peak_gib'],r['eval_peak_gib'])<=70];fastest=max(r['samples_per_second'] for r in valid);chosen=max((r for r in valid if r['samples_per_second']>=.95*fastest),key=lambda r:r['batch_size']);batch=chosen['batch_size'];n=311510;prob=.9;steps=math.ceil(14*n/(batch*prob))
 cfg=yaml.safe_load((ROOT/'config.yaml').read_text());cfg['train'].update(batch_size=batch,accumulation=1,steps=steps,warmup=math.ceil(.5*n/batch),eval_every=math.ceil(n/(prob*batch)));(ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 report.update(selected=chosen,all_trials=rows,effective_batch=batch,steps=steps,expected_total_exposures=steps*batch/n,expected_text_exposures=steps*batch*prob/n,warmup_updates=cfg['train']['warmup'],eval_every_updates=cfg['train']['eval_every']);atomic_json(ROOT/'batch_and_budget.json',report);print(json.dumps(report['selected']),steps,flush=True)
