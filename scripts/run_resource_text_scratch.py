"""Random-initialized generator; only frozen Qwen text features are reused."""
import argparse,hashlib,json,signal,time
from pathlib import Path
import torch
from mcgen.catalog import atomic_json
from mcgen.config import Experiment
from mcgen.training import Trainer
from run_resource_text_training import merged
ROOT=Path('outputs/resource_text_scratch_20261103')
CATALOG=Path('outputs/resource_text_training_20261103/catalog')

def train(device,smoke):
 cfg=Experiment.load(ROOT/'config.yaml');out=ROOT/'training';latest=out/'latest.pt'
 if smoke and latest.exists():raise ValueError('Scratch smoke requires an empty run')
 # No initialize argument: no previous generator checkpoint is loaded.
 trainer=Trainer(cfg,CATALOG,out,device,device,resume=latest if latest.exists() else None,encoder=merged())
 if not latest.exists():
  assert trainer.step==0 and trainer.initialization is None and not trainer.optimizer.state
  digest=hashlib.sha256()
  for name,value in trainer.model.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  atomic_json(ROOT/'initialization_audit.json',{'generator_initialization':'random','generator_checkpoint_loaded':False,'step':trainer.step,'optimizer_states':len(trainer.optimizer.state),'initialization':trainer.initialization,'initial_model_sha256':digest.hexdigest(),'text_encoder':'frozen pretrained Qwen3-8B; exact cached features only'})
  trainer.save('initial_random.pt')
 signal.signal(signal.SIGTERM,lambda *_:setattr(trainer,'stop_requested',True))
 target=20 if smoke else cfg.train.steps;began=time.time();trainer.run(stop_after=target)
 if trainer.step!=target:
  atomic_json(ROOT/'stop_record.json',{'step':trainer.step,'reason':'requested_stop'});raise SystemExit(2)
 if smoke:atomic_json(ROOT/'smoke.json',{'passed':True,'step':trainer.step,'seconds':time.time()-began,'validation_flow':trainer.best,'random_initialization_verified':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train','evaluate']);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4,a.device)
 if a.stage=='evaluate':
  import run_resource_text_training as evaluation
  evaluation.ROOT=ROOT
  evaluation.evaluate(a.device)
 else:train(a.device,a.stage=='smoke')
