"""Measured-batch pooled-embedding training with exposure-based checkpoints and diagnostics."""
import argparse,hashlib,json,signal,time
from pathlib import Path
import torch
from PIL import Image
from mcgen.catalog import atomic_json
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.training import Trainer
from mcgen.tasks import conditioned_prompts
from mcgen.images import decode
from mcgen.sampling import sample
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')

@torch.inference_mode()
def conditioning_diagnostic(trainer):
 trainer.model.eval();gen=torch.Generator().manual_seed(7919);ids=torch.randperm(len(trainer.val_data),generator=gen)[:128].tolist();batch=trainer.val_data.batch(ids,trainer.device)
 text,mask=trainer.encoder.encode(conditioned_prompts(batch['prompts'],trainer.cfg),trainer.device)
 real=trainer.model.context(text,mask,None,None);wrong=trainer.model.context(text.roll(1,0),mask.roll(1,0),None,None);null=trainer.model.context(text,torch.zeros_like(mask),None,None)
 noise=torch.randn(batch['image'].shape,generator=torch.Generator(device=trainer.device).manual_seed(104729),device=trainer.device);target=noise-batch['image'];report={}
 for level in [.1,.5,.9]:
  t=torch.full((len(ids),),level,device=trainer.device);x=(1-level)*batch['image']+level*noise;score={}
  for name,ctx in [('real_text',real),('shuffled_text',wrong),('no_text',null)]:
   with trainer.autocast():v=trainer.model(x,t,ctx)
   score[name]=torch.nn.functional.mse_loss(v.float(),target).item()
  score['gain_over_null']=score['no_text']-score['real_text'];score['gain_over_shuffle']=score['shuffled_text']-score['real_text'];report[str(level)]=score
 trainer.model.train();return {'step':trainer.step,'samples':128,'t_definition':'t=1 is pure noise','noise_level_scores':report}

@torch.inference_mode()
def generate(model,encoder,cfg,out,device,limit=None):
 cases=json.load(open(ROOT/'eval_cases.json'));cases=cases if limit is None else [r for r in cases if r['group']=='novel_composition' and r['seed']==20261112][:limit]
 out.mkdir(parents=True,exist_ok=True);model.eval()
 for start in range(0,len(cases),8):
  batch=cases[start:start+8];text,mask=encoder.encode(conditioned_prompts([r['evaluation_prompt'] for r in batch],cfg),device)
  values=sample(model,text,mask,seeds=[r['seed'] for r in batch],steps=32,text_scale=2,solver='heun')
  for j,v in enumerate(values):Image.fromarray(decode(v)).save(out/f'{start+j:03d}.png')
 atomic_json(out/'cases.json',cases);model.train()

def train(device,smoke):
 cfg=Experiment.load(ROOT/'config.yaml');out=ROOT/'training';checkpoint=out/'latest.pt';encoder=DiskTextEncoder(ROOT/'features')
 trainer=Trainer(cfg,ROOT/'catalog',out,device,device,resume=checkpoint if checkpoint.exists() else None,encoder=encoder)
 signal.signal(signal.SIGTERM,lambda *_:setattr(trainer,'stop_requested',True))
 if trainer.step==0:
  assert trainer.initialization is None and not trainer.optimizer.state
  atomic_json(ROOT/'initialization_audit.json',{'step':0,'generator_weights':'random','optimizer_states':0,'pretrained_generator_loaded':False,'encoder_identity':encoder.identity,'parameters':sum(p.numel() for p in trainer.model.parameters())});trainer.save('initial_random.pt')
 if smoke:
  torch.cuda.reset_peak_memory_stats(device);begin=time.time();trainer.run(stop_after=20)
  assert trainer.step==20
  atomic_json(ROOT/'smoke.json',{'passed':True,'step':trainer.step,'seconds':time.time()-begin,'flow_mse':trainer.best,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3})
  return
 while trainer.step<cfg.train.steps:
  target=min(cfg.train.steps,(trainer.step//cfg.train.eval_every+1)*cfg.train.eval_every);trainer.run(stop_after=target)
  if trainer.step!=target:raise SystemExit(2)
  atomic_json(ROOT/'diagnostics'/f'{trainer.step:06d}.json',conditioning_diagnostic(trainer))
  generate(trainer.model,encoder,cfg,ROOT/'generation_checks'/f'{trainer.step:06d}',device,18)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train','evaluate']);a=p.parse_args();device='cuda:0';torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9,device)
 if a.stage=='evaluate':
  from mcgen.model import PixelFlow
  from mcgen.runtime import load_checkpoint
  cfg=Experiment.load(ROOT/'config.yaml');encoder=DiskTextEncoder(ROOT/'features');model=PixelFlow(cfg.model).to(device)
  for name in ['best_flow','latest']:
   state=load_checkpoint(ROOT/'training'/f'{name}.pt');model.load_state_dict(state['model']);generate(model,encoder,cfg,ROOT/'evaluation'/name,device)
 else:train(device,a.stage=='smoke')
