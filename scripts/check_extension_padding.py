"""Paired 500-update control, same parent Adam state and global RNG, padding only."""
import json,time
from pathlib import Path
import torch
from mcgen.config import Experiment
from mcgen.training import Trainer
from mcgen.runtime import load_checkpoint
from mcgen.catalog import atomic_json
ROOT=Path('outputs/text_expansion22k_20261102');BASE=Path('outputs/text_expansion2k_20261102');OUT=ROOT/'padding_control'
if __name__=='__main__':
 torch.set_num_threads(4);device='cuda:0'
 while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
 config=Experiment.load(ROOT/'config.yaml');trainer=Trainer(config,BASE/'catalog',OUT/'full',device,device,initialize=BASE/'training/latest.pt')
 for name,trim in [('full',False),('trim',True)]:
  state=load_checkpoint(BASE/'training/latest.pt');trainer.model.load_state_dict(state['model']);trainer.optimizer.load_state_dict(state['optimizer']);trainer.step=6000;trainer.best=float('inf');trainer.best_by_task={};trainer.out=OUT/name;trainer.out.mkdir(exist_ok=True);trainer.encoder.trim_padding=trim
  atomic_json(OUT/'status.json',{'stage':name,'start':6000,'stop':6500})
  trainer.run(stop_after=6500)
  print(name,'complete',flush=True)
 a=[json.loads(s) for s in (OUT/'full/metrics.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (OUT/'trim/metrics.jsonl').read_text().splitlines()]
 result={'steps_per_arm':500,'same_parent_optimizer':True,'same_global_rng_steps':True,'same_frozen_encoder_cache_reused':True,'full_mean_loss':sum(r['loss'] for r in a)/len(a),'trim_mean_loss':sum(r['loss'] for r in b)/len(b),'full_final_val':a[-1]['text_flow_mse'],'trim_final_val':b[-1]['text_flow_mse'],'mean_abs_paired_loss_difference':sum(abs(x['loss']-y['loss']) for x,y in zip(a,b))/len(a)}
 atomic_json(OUT/'result.json',result);atomic_json(OUT/'status.json',{'stage':'complete'});print(json.dumps(result),flush=True)
