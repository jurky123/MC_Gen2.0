"""Preserve Adam state and global sample sequence while extending to 22000 steps."""
import json,time,shutil
from pathlib import Path
import numpy as np
import torch,yaml
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.training import Trainer,flow_loss
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts
ROOT=Path('outputs/text_expansion22k_20261102');BASE=Path('outputs/text_expansion2k_20261102')

def main():
 torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
 cfg=yaml.safe_load((BASE/'config.yaml').read_text());cfg['train'].update(steps=22000,schedule_start=6000,warmup=200,lr=3e-5,eval_every=1000)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));config=Experiment.load(ROOT/'config.yaml')
 parent=BASE/'training/latest.pt';state=load_checkpoint(parent);assert state['step']==6000
 atomic_json(ROOT/'protocol.json',{'initialization':str(parent),'initialization_sha256':file_hash(parent),'config_sha256':file_hash(ROOT/'config.yaml'),'catalog':str(BASE/'catalog'),'catalog_fingerprint':state['catalog_fingerprint'],'optimizer_preserved':True,'global_sampling_step_preserved':True,'start_step':6000,'additional_steps':16000,'endpoint':22000,'schedule':'200-step warmup to 3e-5, then cosine to 3e-6 across additional 16000 steps. Old 6000-step run used a different schedule.','caption_policy':'Unchanged target-uniform filename/description mixture and filename-only fallback.','production_promoted':False})
 shutil.copyfile(BASE/'eval_cases.json',ROOT/'eval_cases.json')
 device='cuda:1'
 while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
 trainer=Trainer(config,BASE/'catalog',ROOT/'training',device,device,initialize=parent)
 assert trainer.meta['fingerprint']==state['catalog_fingerprint']
 # Use a real mixed-caption batch to measure cached-text forward/backward work.
 indices,modes=trainer.sampler.sample(16,torch.Generator().manual_seed(20261119));batch=trainer.train_data.batch(indices,device);batch['task_modes']=modes
 full,mask=trainer.encoder.encode(conditioned_prompts(batch['prompts'],config),device);trainer.encoder.trim_padding=True
 short,smask=trainer.encoder.encode(conditioned_prompts(batch['prompts'],config),device)
 assert torch.equal(full[:,:short.shape[1]],short) and torch.equal(mask[:,:short.shape[1]],smask) and not mask[:,short.shape[1]:].any()
 x=batch['image'];g=torch.Generator(device=device).manual_seed(123);noise=torch.randn(x.shape,device=device,generator=g);t=torch.rand(len(x),device=device,generator=g);noisy=(1-t[:,None,None,None])*x+t[:,None,None,None]*noise
 times={False:[],True:[]};losses={False:[],True:[]}
 for trim in [False,True,True,False]:
  trainer.encoder.trim_padding=trim
  for _ in range(3):
   trainer.model.zero_grad(set_to_none=True)
   with trainer.autocast():loss=flow_loss(trainer.model(noisy,t,trainer.condition(batch)),x,noisy,t,noise,batch['tileable'],0)
   loss.backward()
  torch.cuda.synchronize(device);start=time.perf_counter()
  for _ in range(30):
   trainer.model.zero_grad(set_to_none=True)
   with trainer.autocast():loss=flow_loss(trainer.model(noisy,t,trainer.condition(batch)),x,noisy,t,noise,batch['tileable'],0)
   loss.backward()
  torch.cuda.synchronize(device);times[trim].append((time.perf_counter()-start)/30);losses[trim].append(loss.item())
 ratio=float(np.mean(times[False])/np.mean(times[True]));delta=abs(np.mean(losses[False])-np.mean(losses[True]));enabled=bool(ratio>1.03 and delta<1e-3)
 atomic_json(ROOT/'speed_probe.json',{'full_tokens':full.shape[1],'trimmed_tokens':short.shape[1],'valid_features_identical':True,'cached_forward_backward_seconds_full':times[False],'cached_forward_backward_seconds_trimmed':times[True],'speedup':ratio,'loss_abs_difference':delta,'enabled':enabled,'scope':'Two interleaved 30-iteration trials per mode, real batch 16, forward+backward+cached text transfer. Excludes optimizer, validation, initial Qwen encoding and dataset loading. Shared GPU; not end-to-end throughput.'})
 trainer.encoder.trim_padding=enabled;trainer.optimizer.load_state_dict(state['optimizer']);trainer.step=state['step'];trainer.model.zero_grad(set_to_none=True)
 assert all(int(s['step'])==6000 for s in trainer.optimizer.state.values() if 'step' in s)
 run=json.loads((ROOT/'training/run.json').read_text());run['continuation']={'optimizer_preserved':True,'start_step':6000,'trim_padding':enabled};atomic_json(ROOT/'training/run.json',run)
 atomic_json(ROOT/'status.json',{'stage':'training','start_step':6000,'endpoint':22000,'trim_padding':enabled});print('training continuation',json.dumps({'step':trainer.step,'trim_padding':enabled,'speedup':ratio}),flush=True)
 start=time.perf_counter();trainer.run();atomic_json(ROOT/'training_wall_time.json',{'seconds':time.perf_counter()-start,'additional_steps':16000,'includes_encoding_validation_checkpointing':True})
 atomic_json(ROOT/'status.json',{'stage':'training_complete','step':trainer.step})
if __name__=='__main__':main()
