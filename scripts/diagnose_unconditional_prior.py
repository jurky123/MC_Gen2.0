"""Check whether the existing pixel model can learn a 16-item marginal prior."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from mcgen.config import Experiment
from mcgen.training import Trainer
from mcgen.catalog import atomic_json
from mcgen.images import encode,decode,composite
from mcgen.sampling import sample
from audit_item_captions import array

class UnconditionalTrainer(Trainer):
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        batch=self.val_data.batch(list(range(len(self.val_data))),self.device);x=batch['image']
        gen=torch.Generator(device=self.device).manual_seed(104729)
        t=torch.rand(len(x),generator=gen,device=self.device);noise=torch.randn(x.shape,generator=gen,device=self.device)
        # Explicit task modes clear every text/ref mask in the shared condition path.
        batch['task_modes']=['unconditional']*len(x)
        with self.autocast():
            pred=self.model((1-t[:,None,None,None])*x+t[:,None,None,None]*noise,t,self.condition(batch))
        value=float((pred.float()-(noise-x)).square().mean());self.model.train()
        return {'unconditional_flow_mse':value,'unconditional_samples':len(x)}

@torch.no_grad()
def evaluate(trainer,rows,out,label):
    trainer.model.eval();train=[r for r in rows if r['split']=='train'];bank=torch.stack([encode(array(r)) for r in train])
    imgs=[];errors=[];nearest=[]
    for start in range(0,64,8):
        text,mask=trainer.encoder.encode(['']*8,trainer.device);mask.zero_()
        tensors=sample(trainer.model,text,mask,seeds=list(range(100+start,108+start)),steps=32,text_scale=0,solver='heun')
        for tensor in tensors:
            distances=(bank-tensor.cpu()).square().mean((1,2,3));errors.append(float(distances.min()));nearest.append(int(distances.argmin()));imgs.append(decode(tensor))
    sheet=Image.new('RGB',(8*96,8*96),(128,128,128))
    for i,img in enumerate(imgs):
        Image.fromarray(img).save(out/f'{label}_{i:02d}.png')
        sheet.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),((i%8)*96,(i//8)*96))
    sheet.save(out/f'{label}_grid.png');trainer.model.train()
    return {'mean_nearest_train_model_mse':float(np.mean(errors)),'nearest_target_indices':nearest,'unique_nearest_train_targets':len(set(nearest)),'no_foreground':sum(not (a[...,3]>=128).any() for a in imgs),'note':'Nearest-target diagnostics measure fit to this tiny training distribution, not generalization or independent semantic accuracy.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();torch.set_num_threads(4)
    out=a.root/'unconditional_probe';out.mkdir(exist_ok=False);images=out/'images';images.mkdir()
    while torch.cuda.mem_get_info(a.device)[0]<23*1024**3:time.sleep(30)
    free,total=torch.cuda.mem_get_info(a.device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),a.device)
    cfg=Experiment.load(a.root/'config.yaml');cfg.train.task_mix={'text':0.,'reference':0.,'joint':0.,'unconditional':1.}
    cfg.train.steps=800
    rows=json.loads((a.root/'cases.json').read_text())
    trainer=UnconditionalTrainer(cfg,a.root/'catalogs/long',out/'training',a.device,a.device,initialize='outputs/attribute_prior_20261102/text/best_flow.pt')
    result={'before':evaluate(trainer,rows,images,'before')};atomic_json(out/'summary.json',result)
    trainer.run();result['after']=evaluate(trainer,rows,images,'after');atomic_json(out/'summary.json',result)
