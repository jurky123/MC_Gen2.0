"""Bounded pure-resource-text training queue: exact delta cache, smoke, continuation, evaluation."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder,MergedDiskTextEncoder
from mcgen.tasks import conditioned_prompts
ROOT=Path('outputs/resource_text_training_20261103')
BASE=[Path('outputs/fullscale_text10k_20261103/text_features'),Path('outputs/pixel_quality_pilot_20261103/features')]
PARENT=Path('outputs/fullscale_text10k_20261103/training/best_flow.pt')

def features(device):
 from mcgen.conditioning import TextEncoder
 cfg=Experiment.load(ROOT/'config.yaml');base=MergedDiskTextEncoder([DiskTextEncoder(p) for p in BASE])
 prompts=[]
 for line in (ROOT/'catalog/samples.jsonl').open():
  r=json.loads(line);assert r['caption_variant']=='short' and r['caption_supervision']=='deterministic_resource_text_only';prompts.append(r['prompt'])
 prompts.extend(r['evaluation_prompt'] for r in json.load(open(ROOT/'eval_cases.json')));prompts.append('')
 prompts=sorted(set(conditioned_prompts(prompts,cfg)));missing=[p for p in prompts if p not in base.index]
 out=ROOT/'delta_features';out.mkdir(exist_ok=True);shape=[len(missing),cfg.model.text_length,cfg.model.text_dim]
 atomic_json(ROOT/'cache_plan.json',{'required_prompts':len(prompts),'reused':len(prompts)-len(missing),'new':len(missing),'new_bytes':int(np.prod(shape))*2,'base_caches':[str(p) for p in BASE],'text_identity':base.identity})
 if (out/'metadata.json').exists():
  meta=json.load(open(out/'metadata.json'))
  if meta['complete']:
   DiskTextEncoder(out,base.identity);return
 if not (out/'prompts.json').exists():atomic_json(out/'prompts.json',missing)
 assert json.load(open(out/'prompts.json'))==missing
 atomic_json(out/'metadata.json',{'complete':False,'shape':shape,'text_identity':base.identity,'prompts_sha256':file_hash(out/'prompts.json')})
 if not missing:raise ValueError('No delta features needed; use source-only encoder')
 mode='r+' if (out/'features.uint16.mmap').exists() else 'w+'
 f=np.memmap(out/'features.uint16.mmap',dtype='uint16',mode=mode,shape=tuple(shape));m=np.memmap(out/'masks.uint8.mmap',dtype='uint8',mode=mode,shape=tuple(shape[:2]))
 start=json.load(open(out/'progress.json'))['done'] if (out/'progress.json').exists() else 0
 encoder=TextEncoder(cfg.text,cfg.model,device);assert encoder.identity==base.identity
 began=time.time()
 for i in range(start,len(missing),16):
  h,mask=encoder.encode(missing[i:i+16],'cpu');f[i:i+len(h)]=h.view(torch.uint16).numpy();m[i:i+len(h)]=mask.numpy();f.flush();m.flush()
  atomic_json(out/'progress.json',{'done':i+len(h),'total':len(missing),'seconds_this_run':time.time()-began})
  if i%256==0:print('cached',i+len(h),len(missing),round(time.time()-began,1),flush=True)
 # Original feature batches are recomputed before commit; prompt order remains exact.
 for i in [0,max(0,(len(missing)//2)//16*16),max(0,(len(missing)-1)//16*16)]:
  h,mask=encoder.encode(missing[i:i+16],'cpu');assert np.array_equal(f[i:i+len(h)],h.view(torch.uint16).numpy());assert np.array_equal(m[i:i+len(h)],mask.numpy())
 atomic_json(out/'metadata.json',{'complete':True,'shape':shape,'text_identity':base.identity,'prompts_sha256':file_hash(out/'prompts.json')})
 atomic_json(out/'verification.json',{'exact_bf16_recomputed':True,'positions':'first/middle/last original16-row batches','text_identity':base.identity})

def merged():return MergedDiskTextEncoder([DiskTextEncoder(p) for p in BASE+[ROOT/'delta_features']])

def train(device,smoke=False):
 from mcgen.training import Trainer
 cfg=Experiment.load(ROOT/'config.yaml');report=json.load(open(ROOT/'build_report.json'));assert file_hash(PARENT)==report['parent_sha256']
 out=ROOT/'training';latest=out/'latest.pt';encoder=merged()
 trainer=Trainer(cfg,ROOT/'catalog',out,device,device,resume=latest if latest.exists() else None,initialize=None if latest.exists() else PARENT,encoder=encoder)
 signal.signal(signal.SIGTERM,lambda *_:setattr(trainer,'stop_requested',True))
 target=20 if smoke else cfg.train.steps;trainer.run(stop_after=target)
 if smoke:assert trainer.step==20;atomic_json(ROOT/'smoke.json',{'passed':True,'step':trainer.step,'validation_flow':trainer.best,'visual_labels_used':False})
 else:assert trainer.step==10000


def evaluate(device):
 from PIL import Image
 from mcgen.model import PixelFlow
 from mcgen.runtime import load_checkpoint
 from mcgen.sampling import sample
 from mcgen.images import decode
 from evaluate_text_recovery import color_fraction
 cfg=Experiment.load(ROOT/'config.yaml');encoder=merged();cases=json.load(open(ROOT/'eval_cases.json'));out=ROOT/'evaluation';out.mkdir(exist_ok=True);model=PixelFlow(cfg.model).to(device);results=[]
 for arm,path in {'before':PARENT,'best':ROOT/'training/best_flow.pt','latest':ROOT/'training/latest.pt'}.items():
  state=load_checkpoint(path);model.load_state_dict(state['model']);model.eval();del state;(out/arm).mkdir(exist_ok=True)
  for i in range(0,len(cases),8):
   batch=cases[i:i+8];text,mask=encoder.encode(conditioned_prompts([r['evaluation_prompt'] for r in batch],cfg),device)
   with torch.inference_mode():raw=sample(model,text,mask,seeds=[r['seed'] for r in batch],steps=32,text_scale=2,solver='heun')
   for j,(r,v) in enumerate(zip(batch,raw)):
    image=decode(v);Image.fromarray(image).save(out/arm/f'{i+j:03d}.png');results.append({'arm':arm,'case':i+j,'requested_color_fraction':color_fraction(image,r['requested_color']) if r.get('requested_color') else None})
 atomic_json(out/'results.json',results)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','smoke','train','evaluate']);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4,a.device)
 if a.stage=='features':features(a.device)
 elif a.stage in ['smoke','train']:train(a.device,a.stage=='smoke')
 else:evaluate(a.device)
