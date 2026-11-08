"""Two matched16-target fitting experiments, not a production training run."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.curated_training import MixedData
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.overfit_diagnostics import (recover_target_pixels, same_category_permutation,
                                      select_training_targets, target_pixel_tokens)
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from train_curated_mixed import loss_for, optimizer

SOURCE=Path('outputs/reference_native16_lossless_20261107')
SEEDS=(2026110801,2026110802)
CHECKPOINTS=(0,250,1000,2000)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    (root/'inputs').mkdir()
    (root/'generated').mkdir()
    data=MixedData(SOURCE)
    chosen=select_training_targets(data.rows)
    indices=[i for i,_ in chosen]
    categories=[c for _,c in chosen]
    rows=[{**data.rows[i],'case_id':f'{category}_{j%4}','pair_index':i,'fixed_prompt':category}
          for j,(i,category) in enumerate(chosen)]
    for row in rows:
        for key,suffix in [('reference_file','reference'),('target_file','target')]:
            with Image.open(row[key]) as im:
                im.convert('RGBA').save(root/'inputs'/f'{row["case_id"]}_{suffix}.png')
    text,mask=data.encoder.encode(conditioned_prompts(categories,data.cfg),'cpu')
    target=data.targets[indices].clone()
    oracle=target_pixel_tokens(target)
    assert torch.equal(recover_target_pixels(oracle),target)
    normalized=torch.nn.functional.layer_norm(oracle,(4096,))
    error=(recover_target_pixels(normalized,normalized=True)-target).abs().max().item()
    if error>1e-6:raise ValueError('Oracle control loses information in token LayerNorm')
    features={'target':target,'text':text,'mask':mask,
        'qwen_hd':torch.from_numpy(np.array(data.features[indices],copy=True)).view(torch.bfloat16),
        'oracle_target':oracle,'same_category_permutation':same_category_permutation(categories),
        'text_identity':data.encoder.identity,'visual_identity':data.visual_identity}
    torch.save(features,root/'features.pt')
    atomic_json(root/'samples.json',rows)
    checkpoint=SOURCE/'training/best_joint_flow.pt'
    atomic_json(root/'protocol.json',{'source_run':str(SOURCE.resolve()),'source_checkpoint':str(checkpoint.resolve()),
        'source_checkpoint_sha256':file_hash(checkpoint),'source_pairs_sha256':data.pairs_sha256,
        'features_sha256':file_hash(root/'features.pt'),'samples_sha256':file_hash(root/'samples.json'),
        'targets':16,'selection':'4 distinct HQ training targets each: helmet, sword, axe/battleaxe, shield; deterministic seed20261108',
        'fixed_category_text':True,'unique_item_names_used_as_conditions':False,
        'train_steps':2000,'batch_size':16,'task_mix':{'fixed_category_and_reference':.8,'reference_only':.2},
        'learning_rates':{'generator':1e-5,'visual_adapter':1e-4},'warmup_steps':50,
        'schedule':'constant after warmup; no early cosine decay','trainable':'same full generator and visual adapter as current mixed training',
        'objective':'unchanged uniform-time Flow Matching MSE','precision':'BF16 training, FP32 generation',
        'paired_randomness':'identical batch permutations, text-presence draws, times and Gaussian training noise across arms',
        'generation_seeds':SEEDS,'sampling_steps':32,'solver':'heun','primary_cfg':1.,'secondary_cfg':2.,
        'generation_noise':'same CPU iid16 Gaussian for every design, shared across both arms; no target-dependent seeds',
        'arms':{'qwen_hd':'existing frozen Qwen spatial features of actual HD references',
                'oracle_target':'invertible2x2 target-pixel packing,64x4096 with fixed anchors; target leakage, diagnostic upper bound only'},
        'oracle_layernorm_max_recovery_error':error,'optimizer_reset':True,'production_checkpoint_overwritten':False,
        'scope':'training-set fitting only; no generalization claim; oracle not deployable'})
    print('PREPARED16 training targets; oracle normalized recovery error',error,flush=True)


def image_scores(prediction,target,categories):
    # Both tensors use the same premultiplied training representation.
    p=(prediction.detach().float().cpu().clamp(-1,1)+1)/2
    q=(target.detach().float().cpu()+1)/2
    errors=(p-q).abs()
    scores=[]
    for i,category in enumerate(categories):
        a,b=p[i,3]>=.5,q[i,3]>=.5
        union=a|b
        choices=[j for j,c in enumerate(categories) if c==category]
        distances=[(p[i]-q[j]).abs().mean().item() for j in choices]
        recovered=choices[int(np.argmin(distances))]
        scores.append({'case_index':i,'premultiplied_mae':errors[i].mean().item(),
            'alpha_iou':((a&b).sum()/union.sum()).item() if union.any() else 1.,
            'foreground_mae':errors[i][:,union].mean().item() if union.any() else 0.,
            'retrieved_target_index':recovered,'correct_design':recovered==i})
    return {'premultiplied_mae':float(np.mean([s['premultiplied_mae'] for s in scores])),
            'alpha_iou':float(np.mean([s['alpha_iou'] for s in scores])),
            'foreground_mae':float(np.mean([s['foreground_mae'] for s in scores])),
            'correct_designs':sum(s['correct_design'] for s in scores),'cases':scores}


@torch.inference_mode()
def evaluate(model,values,rows,root,arm,step,device,final=False):
    directory=root/arm/f'eval_{step:06d}'
    directory.mkdir(parents=True,exist_ok=True)
    target=values['target'].to(device)
    text=values['text'].to(device)
    mask=values['mask'].to(device)
    refs=values[arm].to(device)
    permutation=values['same_category_permutation'].to(device)
    categories=[r['fixed_prompt'] for r in rows]
    report={'step':step,'arm':arm,'generation':{},'flow':{}}
    for level in (.1,.5,.9,.99):
        g=torch.Generator(device=device).manual_seed(9901+int(level*100))
        noise=torch.randn(target.shape,generator=g,device=device)
        x=(1-level)*target+level*noise
        errors={}
        for label,reference in [('matched',refs),('wrong_same_category',refs[permutation]),('no_reference',None)]:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(x,torch.full((len(rows),),level,device=device),model.context(text,mask,reference,None))
            errors[label]=(prediction.float()-(noise-target)).square().mean().item()
        report['flow'][str(level)]=errors
    tasks=[('joint',mask,refs,1.),('reference',torch.zeros_like(mask),refs,1.),('text',mask,None,1.)]
    if final:
        tasks += [('wrong_reference',mask,refs[permutation],1.),('joint_cfg2',mask,refs,2.),
                  ('reference_cfg2',torch.zeros_like(mask),refs,2.)]
    for si,seed in enumerate(SEEDS if final else SEEDS[:1]):
        noise=torch.randn((1,4,16,16),generator=torch.Generator().manual_seed(seed)).repeat(len(rows),1,1,1)
        for task,active_mask,reference,cfg in tasks:
            prediction=sample(model,text,active_mask,reference,seeds=[seed]*len(rows),steps=32,
                text_scale=cfg,ref_scale=cfg,initial_noise=noise)
            if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite sampled image')
            actual_target=target[permutation] if task=='wrong_reference' else target
            scores=image_scores(prediction,actual_target,categories)
            report['generation'][f'{task}_s{si}']=scores
            for j,row in enumerate(rows):
                name=f'{arm}_{step:06d}_{row["case_id"]}_{task}_s{si}.png'
                Image.fromarray(decode(prediction[j])).save(root/'generated'/name)
    atomic_json(directory/'metrics.json',report)
    with (root/arm/'evaluations.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    print('EVAL',arm,step,json.dumps({k:{s:v[s] for s in ('premultiplied_mae','alpha_iou','correct_designs')}
          for k,v in report['generation'].items()}),flush=True)
    return report


def train(root,arm,device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.30,device)
    protocol=json.loads((root/'protocol.json').read_text())
    if file_hash(root/'features.pt')!=protocol['features_sha256'] or file_hash(root/'samples.json')!=protocol['samples_sha256']:
        raise ValueError('Diagnostic inputs changed')
    values=torch.load(root/'features.pt',weights_only=True)
    rows=json.loads((root/'samples.json').read_text())
    if any(r['split']!='train' for r in rows):raise ValueError('Only training targets allowed')
    if file_hash(protocol['source_checkpoint'])!=protocol['source_checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    state=load_checkpoint(protocol['source_checkpoint'])
    if state['text_identity']!=values['text_identity'] or state['visual_identity']!=values['visual_identity']:
        raise ValueError('Encoder identity changed')
    cfg=Experiment.from_dict(state['config'])
    cfg.train.steps=protocol['train_steps']
    cfg.train.batch_size=protocol['batch_size']
    cfg.train.warmup=protocol['warmup_steps']
    cfg.train.task_mix={'joint':.8,'reference':.2,'text':0.,'unconditional':0.}
    torch.manual_seed(20261108)
    model=PixelFlow(cfg.model).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    initial_digest=hashlib.sha256()
    for k,v in sorted(state['model'].items()):
        initial_digest.update(k.encode());initial_digest.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    del state
    gc.collect()
    opt=optimizer(model,1e-5)
    directory=root/arm;directory.mkdir(exist_ok=True)
    step=0;best=float('inf')
    if (directory/'latest.pt').exists():
        saved=load_checkpoint(directory/'latest.pt')
        if saved['arm']!=arm or saved['protocol_sha256']!=file_hash(root/'protocol.json'):
            raise ValueError('Resume contract changed')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        step=saved['step'];best=saved['best_generation_mae']
        del saved
    atomic_json(directory/'run.json',{'arm':arm,'initial_model_sha256':initial_digest.hexdigest(),
        'protocol_sha256':file_hash(root/'protocol.json'),'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'initialization':'current native16 best,step270; optimizer reset'})
    def save(name,include_optimizer=True):
        payload={'config':cfg.to_dict(),'model':model.state_dict(),'step':step,'arm':arm,
                 'protocol_sha256':file_hash(root/'protocol.json'),'best_generation_mae':best,
                 'text_identity':values['text_identity'],
                 'visual_identity':(values['visual_identity'] if arm=='qwen_hd' else
                                    hashlib.sha256(b'diagnostic_target_pixel_tokens_v1').hexdigest()),
                 'source_visual_identity':values['visual_identity'],
                 'visual_feature_protocol':'qwen_hd_spatial' if arm=='qwen_hd' else 'diagnostic_target_pixel_tokens_v1',
                 'diagnostic_only':True,'target_leakage':arm=='oracle_target'}
        if include_optimizer:payload['optimizer']=opt.state_dict()
        save_checkpoint(directory/name,payload)
    target=values['target'].to(device)
    text=values['text'].to(device)
    refs=values[arm].to(device)
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    started=time.monotonic()
    if step==0:
        initial=evaluate(model,values,rows,root,arm,0,device)
        best=initial['generation']['joint_s0']['premultiplied_mae']
    for update in range(step,protocol['train_steps']):
        if stopping[0]:break
        rng=torch.Generator().manual_seed(20261108+update)
        ids=torch.randperm(len(rows),generator=rng).to(device)
        present_text=(torch.rand(len(rows),generator=rng)>=.2).to(device)
        mask=present_text[:,None].expand(-1,text.shape[1])
        batch=(target[ids],text[ids],mask,refs[ids],torch.ones(len(rows),dtype=torch.bool,device=device))
        warm=min(1.,(update+1)/50)
        for group in opt.param_groups:group['lr']=1e-5*warm*group['lr_factor']
        opt.zero_grad(set_to_none=True)
        loss,_=loss_for(model,batch,device,20261108+update+100000)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        extra={}
        if update%100==0:
            for prefix,label in [('visual_adapter.','visual'),('','all')]:
                grads=[p.grad.detach().float().square().sum() for name,p in model.named_parameters()
                       if name.startswith(prefix) and p.grad is not None]
                extra[label+'_gradient_norm']=torch.stack(grads).sum().sqrt().item()
            extra['gate_mean_abs']={k:v.detach().tanh().abs().mean().item() for k,v in model.visual_adapter.gates.items()}
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();step=update+1
        metric={'step':step,'loss':loss.item(),'gradient_norm':norm.item(),'base_lr':opt.param_groups[0]['lr'],
                'seconds':time.monotonic()-started,**extra}
        with (directory/'training.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        if step%100==0:print('STEP',arm,step,round(loss.item(),6),flush=True)
        if step in CHECKPOINTS[1:]:
            result=evaluate(model,values,rows,root,arm,step,device,final=step==protocol['train_steps'])
            score=result['generation']['joint_s0']['premultiplied_mae']
            if score<best:
                best=score;save('best_generation.pt',False)
            save('latest.pt')
        elif step%250==0:save('latest.pt')
        atomic_json(directory/'status.json',{'phase':'training','step':step,'planned_steps':protocol['train_steps'],
            'complete':False,'loss':loss.item(),'seconds':time.monotonic()-started,'best_generation_mae':best})
    save('latest.pt')
    atomic_json(directory/'status.json',{'phase':'complete' if step==protocol['train_steps'] else 'stopped',
        'step':step,'complete':step==protocol['train_steps'],'seconds':time.monotonic()-started,
        'best_generation_mae':best,'source_checkpoint_unchanged':file_hash(protocol['source_checkpoint'])==protocol['source_checkpoint_sha256']})
    print('FINISHED',arm,step,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train'])
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_overfit_20261108'))
    parser.add_argument('--arm',choices=['qwen_hd','oracle_target'],default='qwen_hd')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.out)
    else:train(args.out,args.arm,args.device)
