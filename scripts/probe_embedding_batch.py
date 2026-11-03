"""One isolated, real-data optimizer/memory/throughput trial for the larger model."""
import argparse,json,time
from pathlib import Path
import torch
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.catalog import atomic_json
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9,a.device)
 cfg=Experiment.load(ROOT/'config.yaml');out=ROOT/'batch_probe';out.mkdir(exist_ok=True);report={'batch_size':a.batch,'device':a.device}
 try:
  data=torch.load(ROOT/'probe_data.pt',map_location='cpu',weights_only=True);ids=torch.arange(a.batch)%len(data['image']);image=data['image'][ids].to(a.device);text=data['text'][ids].to(a.device);mask=data['mask'][ids].to(a.device);mask[::10]=False
  model=PixelFlow(cfg.model).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay);report['parameters']=sum(p.numel() for p in model.parameters())
  noise=torch.randn_like(image);t=torch.rand(a.batch,device=a.device);noisy=(1-t[:,None,None,None])*image+t[:,None,None,None]*noise
  def update():
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(noisy,t,model.context(text,mask,None,None));loss=torch.nn.functional.mse_loss(pred.float(),noise-image)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite probe loss')
  for _ in range(2):update()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(a.device);begin=time.perf_counter()
  for _ in range(3):update()
  torch.cuda.synchronize();seconds=(time.perf_counter()-begin)/3
  train_peak=torch.cuda.max_memory_allocated(a.device);reserved=torch.cuda.max_memory_reserved(a.device)
  model.eval();torch.cuda.reset_peak_memory_stats(a.device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(noisy,t,model.context(text,mask,None,None))
  torch.cuda.synchronize();eval_peak=torch.cuda.max_memory_allocated(a.device)
  report.update(status='ok',seconds_per_update=seconds,samples_per_second=a.batch/seconds,train_peak_gib=train_peak/1024**3,eval_peak_gib=eval_peak/1024**3,reserved_gib=reserved/1024**3)
 except torch.OutOfMemoryError:report.update(status='oom')
 atomic_json(out/f'batch_{a.batch}.json',report);print(json.dumps(report),flush=True)
