"""Conservative continuation: preserve Adam and terminal LR, full padded features."""
import argparse,json,time,shutil
from pathlib import Path
import torch,yaml
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.training import Trainer
from mcgen.runtime import load_checkpoint
ROOT=Path('outputs/text_expansion22k_conservative_20261102');BASE=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 parent=BASE/'training/latest.pt'
 if not a.resume:
  ROOT.mkdir(exist_ok=False);cfg=yaml.safe_load((BASE/'config.yaml').read_text());cfg['train'].update(steps=22000,schedule_start=6000,warmup=0,lr=5e-6,eval_every=1000)
  (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));shutil.copyfile(BASE/'eval_cases.json',ROOT/'eval_cases.json')
  atomic_json(ROOT/'protocol.json',{'initialization':str(parent),'initialization_sha256':file_hash(parent),'endpoint':22000,'start_step':6000,'optimizer_preserved':True,'global_sampling_step_preserved':True,'catalog':str(BASE/'catalog'),'config_sha256':file_hash(ROOT/'config.yaml'),'schedule':'Continue at prior terminal LR 5e-6, cosine decay to 5e-7 over additional 16000 steps; no LR restart or warmup.','trim_padding':False,'scope':'Conservative replacement after aggressive LR/padding experiment regressed. Same images, caption policy, optimizer and RNG progression.','production_promoted':False})
 device='cuda:1'
 while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
 cfg=Experiment.load(ROOT/'config.yaml')
 trainer=Trainer(cfg,BASE/'catalog',ROOT/'training',device,device,**({'resume':ROOT/'training/latest.pt'} if a.resume else {'initialize':parent}))
 if not a.resume:
  state=load_checkpoint(parent);assert state['step']==6000 and trainer.meta['fingerprint']==state['catalog_fingerprint'];trainer.optimizer.load_state_dict(state['optimizer']);trainer.step=6000
 trainer.encoder.trim_padding=False
 assert all(int(s['step'])==trainer.step for s in trainer.optimizer.state.values() if 'step' in s)
 atomic_json(ROOT/'status.json',{'stage':'training','start_step':trainer.step,'endpoint':22000,'trim_padding':False})
 start=time.perf_counter();trainer.run(stop_after=a.stop_after)
 atomic_json(ROOT/'status.json',{'stage':'training_complete' if trainer.step==22000 else 'checkpoint_ready','step':trainer.step,'elapsed_seconds':time.perf_counter()-start})
