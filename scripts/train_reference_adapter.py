"""Train only a residual condition MLP; original text generator stays immutable."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import torch.nn.functional as F
import yaml

from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import content_hash,encode,decode
from mcgen.reference_adapter import initialize_adapter,backbone_digest
from mcgen.runtime import load_checkpoint,save_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts


def data(root,partial=False):
    rows=json.load(open(root/'pairs.json'));paths=sorted((root/'features').glob('*.pt'))
    meta=None
    if not partial:
        meta=json.load(open(root/'features/metadata.json'))
        assert meta['complete'] and meta['pairs_sha256']==file_hash(root/'pairs.json')
        for p in paths:
            if file_hash(p)!=meta['chunks'][p.name]:raise ValueError('changed image cache')
    else:paths=paths[:1]
    if not paths:raise ValueError('no embedding cache chunks')
    chunks=[torch.load(p,weights_only=True) for p in paths]
    identity=chunks[0]['multimodal_identity']
    for p,c in zip(paths,chunks):
        assert c['start']==int(p.stem) and c['multimodal_identity']==identity and c['pairs_sha256']==file_hash(root/'pairs.json')
    image=torch.cat([c['image'] for c in chunks]);joint=torch.cat([c['joint'] for c in chunks]);rows=rows[:len(image)]
    if not partial:
        assert len(rows)==meta['rows'] and len(image)==len(json.load(open(root/'pairs.json')))
    maps={};targets=[]
    for r in rows:
        s=r['target'];key=s['path']
        if key not in maps:maps[key]=np.memmap(key,mode='r',dtype='uint8',shape=tuple(s['shape']))
        a=np.array(maps[key][s['index']]);assert content_hash(a)==r['target_hash'];targets.append(encode(a))
    groups={}
    for split in ('train','val'):
        g=defaultdict(list)
        for i,r in enumerate(rows):
            if r['split']==split:g[r['target_hash']].append(i)
        groups[split]=list(g.values())
    return rows,torch.stack(targets),image,joint,groups,identity


def draw(groups,n,rng):
    selected=torch.randint(len(groups),(n,),generator=rng).tolist()
    return [groups[i][int(torch.randint(len(groups[i]),(),generator=rng))] for i in selected]


def objective(model,x,h,device,seed):
    gen=torch.Generator(device=device).manual_seed(seed)
    t=torch.rand(len(x),generator=gen,device=device);noise=torch.randn(x.shape,generator=gen,device=device)
    noisy=(1-t[:,None,None,None])*x+t[:,None,None,None]*noise
    with torch.autocast('cuda',dtype=torch.bfloat16):
        present=torch.ones(len(x),dtype=torch.bool,device=device)
        ctx=model.context(h,torch.ones(h.shape[:2],dtype=torch.bool,device=device),None,present)
        predicted=model(noisy,t,ctx)
        loss=F.mse_loss(predicted.float(),noise-x)
    return loss


def render_checks(directory,panel,step):
    baseline=Path('outputs/zero_shot_reference_20261104')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    labels=['HD reference','MC target','Text baseline','Zero-shot image','Adapted image','Zero-shot joint','Adapted joint']
    for page in range(4):
        canvas=Image.new('RGB',(1020,1080),'#111827');draw_canvas=ImageDraw.Draw(canvas)
        draw_canvas.text((18,14),f'Reference adapter step {step} | original text backbone frozen | {panel[page*6]["split"].upper()}',font=font,fill='white')
        for col,label in enumerate(labels):draw_canvas.text((18+col*142,42),label,font=font,fill='#e5e7eb')
        for row,case in enumerate(panel[page*6:page*6+6]):
            ident=case['case_id'];y=70+row*167
            paths=[baseline/'inputs'/f'{ident}_reference.png',baseline/'inputs'/f'{ident}_target.png',
                   baseline/'native'/f'{ident}_text_cfg2_seed20261104.png',baseline/'native'/f'{ident}_image_cfg2_seed20261104.png',
                   directory/f'{ident}_reference.png',baseline/'native'/f'{ident}_joint_cfg2_seed20261104.png',directory/f'{ident}_joint.png']
            for col,path in enumerate(paths):
                with Image.open(path) as im:
                    im=im.convert('RGBA');bg=Image.new('RGBA',im.size,(239,241,245,255));bg.alpha_composite(im)
                    picture=bg.convert('RGB').resize((128,128),Image.Resampling.LANCZOS if col==0 else Image.Resampling.NEAREST)
                canvas.paste(picture,(18+col*142,y))
            draw_canvas.text((18,y+133),f'{ident}: {case["prompt"]}',font=font,fill='#e5e7eb')
        canvas.save(directory/f'comparison_{page+1}.png')


def probe(root,device):
    cfg=Experiment.load(root/'config.yaml');audit=json.load(open(root/'data_audit.json'))
    state=load_checkpoint(audit['base_checkpoint']);torch.manual_seed(cfg.train.seed)
    model=initialize_adapter(cfg,state,device);opt=torch.optim.AdamW(model.image_adapter.parameters(),lr=cfg.train.lr)
    rows,x,image,joint,groups,identity=data(root,partial=True);results=[]
    for batch in (64,128,256,384,512):
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);times=[]
            for k in range(3):
                ids=draw(groups['train'],batch,torch.Generator().manual_seed(k));opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device);start=time.time()
                loss=objective(model,x[ids].to(device),image[ids].to(device),device,7919+k);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(),1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize(device);times.append(time.time()-start)
            row={'batch_size':batch,'seconds_per_step':sum(times[1:])/2,'images_per_second':batch/(sum(times[1:])/2),
                 'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3,'loss':loss.item()}
            results.append(row);print('PROBE',row,flush=True)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();results.append({'batch_size':batch,'oom':True});break
    valid=[r for r in results if not r.get('oom')]
    if not valid:raise RuntimeError('no feasible batch')
    chosen=max(valid,key=lambda r:r['images_per_second']);batch=chosen['batch_size']
    # Preserve enough optimizer updates while giving each canonical target
    # roughly 160 conditional draws across both generators and both modes.
    targets=audit['unique_targets']['train'];cfg.train.batch_size=batch
    cfg.train.steps=max(1500,min(4000,math.ceil(targets*160/batch)))
    cfg.train.eval_every=max(100,math.ceil(targets*16/batch))
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
    atomic_json(root/'batch_budget.json',{'measurements':results,'chosen':chosen,'steps':cfg.train.steps,
                 'estimated_training_seconds':cfg.train.steps*chosen['seconds_per_step'],
                 'conditional_draws_per_target':cfg.train.steps*batch/targets,'image_fraction':.5,'joint_fraction':.5,
                 'optimizer_updates_preserved':True,'training_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})


def smoke(root,device):
    cfg=Experiment.load(root/'config.yaml');audit=json.load(open(root/'data_audit.json'))
    base=load_checkpoint(audit['base_checkpoint']);torch.manual_seed(cfg.train.seed)
    model=initialize_adapter(cfg,base,device);before=backbone_digest(model)
    rows,x,image,joint,groups,identity=data(root,partial=True)
    ids=draw(groups['train'],16,torch.Generator().manual_seed(991));target=x[ids].to(device);h=image[ids].to(device)
    opt=torch.optim.AdamW(model.image_adapter.parameters(),lr=cfg.train.lr)
    original=objective(model,target,h,device,1217).item();fresh_before=objective(model,target,h,device,2293).item()
    start=time.time()
    for step in range(100):
        opt.zero_grad(set_to_none=True);loss=objective(model,target,h,device,1217+step);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%20==19:print('SMOKE',step+1,loss.item(),flush=True)
    after=objective(model,target,h,device,1217).item();fresh_after=objective(model,target,h,device,2293).item()
    unchanged=backbone_digest(model)==before
    report={'passed':unchanged and after<original and fresh_after<fresh_before,'updates':100,'evaluation_noise0_loss_before':original,'evaluation_noise0_loss_after':after,
            'fresh_noise_loss_before':fresh_before,'fresh_noise_loss_after':fresh_after,'backbone_exact':unchanged,
            'seconds':time.time()-start,'weights_reused_for_full_run':False,'training_noise':'fresh each update','purpose':'small real-pair gradient/learning check, not held-out image generalization'}
    atomic_json(root/'smoke.json',report);print('SMOKE RESULT',report,flush=True)
    if not report['passed']:raise RuntimeError('real-data adapter smoke failed')


def train(root,device,stop_after=None):
    print('TRAIN SETUP',str(root),str(device),flush=True)
    cfg=Experiment.load(root/'config.yaml');audit=json.load(open(root/'data_audit.json'))
    if file_hash(audit['base_checkpoint'])!=audit['base_checkpoint_sha256']:raise ValueError('changed base checkpoint')
    base=load_checkpoint(audit['base_checkpoint']);torch.manual_seed(cfg.train.seed)
    model=initialize_adapter(cfg,base,device);baseline=backbone_digest(model)
    rows,x,image,joint,groups,identity=data(root)
    print('CACHE VERIFIED',len(rows),flush=True)
    meta=json.load(open(root/'features/metadata.json'));assert meta['text_identity']==base['text_identity']
    opt=torch.optim.AdamW(model.image_adapter.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
    out=root/'training';out.mkdir(exist_ok=True);step=0;best=float('inf');best_modes={}
    if (out/'latest.pt').exists():
        state=load_checkpoint(out/'latest.pt')
        if Experiment.from_dict(state['config']).to_dict()!=cfg.to_dict() or state['multimodal_identity']!=identity or state['catalog_fingerprint']!=audit['pairs_sha256']:raise ValueError('resume contract changed')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];best_modes=state['best_by_task']
    assert backbone_digest(model)==baseline
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    frozen_text=DiskTextEncoder('outputs/qwen_vl_embedding_large_20261103/features',expected_identity=base['text_identity'])
    panel=json.load(open('outputs/zero_shot_reference_20261104/cases.json'))
    panel_lookup={(r['target_hash'],r['reference_generator']):i for i,r in enumerate(rows)}
    panel_ids=[panel_lookup[(r['target_hash'],r['reference_generator'])] for r in panel]
    prompts=[r['prompt'] for r in panel]
    text,mask=frozen_text.encode(conditioned_prompts(prompts,cfg),device)
    text_seeds=[20261104]*len(panel)
    with torch.inference_mode():original_text=sample(model,text,mask,seeds=text_seeds,steps=16,text_scale=2).cpu()
    control_meta=json.load(open(root/'validation_controls.json'))
    if file_hash(root/'validation_controls.pt')!=control_meta['sha256']:raise ValueError('changed validation control')
    controls=torch.load(root/'validation_controls.pt',weights_only=True)
    if controls['multimodal_identity']!=identity:raise ValueError('wrong control encoder protocol')
    val_ids=controls['ids']
    val_text,val_mask=frozen_text.encode(conditioned_prompts([rows[i]['prompt'] for i in val_ids],cfg),device)
    atomic_json(root/'run.json',{'config':cfg.to_dict(),'base_checkpoint':audit['base_checkpoint'],'base_sha256':audit['base_checkpoint_sha256'],
                 'base_step':base['step'],'backbone_sha256':baseline,'text_identity':base['text_identity'],'multimodal_identity':identity,
                 'trainable':'image_adapter only','parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'caption_policy':'cleaned filename supervision; image-only input has no caption',
                 'task_sampling':'uniform target then original-HD generator variant, 50% image-only / 50% image+text',
                 'protection':'No-reference calls bypass the adapter; backbone tensors and fixed text generations must stay exactly unchanged.'})
    start=time.time()
    def save(name):
        save_checkpoint(out/name,{'config':cfg.to_dict(),'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
                'best':best,'best_by_task':best_modes,'catalog_fingerprint':audit['pairs_sha256'],'text_identity':base['text_identity'],
                'multimodal_identity':identity,'initialization':{'path':audit['base_checkpoint'],'sha256':audit['base_checkpoint_sha256'],'step':base['step']},
                'backbone_sha256':baseline})
    @torch.no_grad()
    def validate():
        nonlocal best,best_modes
        if backbone_digest(model)!=baseline:raise RuntimeError('frozen backbone changed')
        actual=sample(model,text,mask,seeds=text_seeds,steps=16,text_scale=2).cpu()
        if not torch.equal(actual,original_text):raise RuntimeError('original text generation changed')
        scores={'step':step,'text_generation_exact':True,'backbone_exact':True}
        for mode,features in [('reference',image),('joint',joint)]:
            h=features[val_ids].to(device);target=x[val_ids].to(device);values=[]
            for k,level in enumerate((.1,.5,.9)):
                gen=torch.Generator(device=device).manual_seed(104729+k);noise=torch.randn(target.shape,generator=gen,device=device)
                noisy=target*(1-level)+noise*level;t=torch.full((len(target),),level,device=device)
                parts={}
                wrong=(image[controls['wrong_ids']] if mode=='reference' else controls['wrong_image_same_text']).to(device)
                for label,condition in [('matched',h),('wrong_reference',wrong)]:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        ctx=model.context(condition,torch.ones(condition.shape[:2],device=device,dtype=torch.bool),None,torch.ones(len(condition),device=device,dtype=torch.bool))
                        prediction=model(noisy,t,ctx)
                    parts[label]=F.mse_loss(prediction.float(),noise-target).item()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    baseline_mask=val_mask if mode=='joint' else torch.zeros_like(val_mask)
                    prediction=model(noisy,t,model.context(val_text,baseline_mask,None,None))
                parts['text_only' if mode=='joint' else 'unconditional']=F.mse_loss(prediction.float(),noise-target).item()
                parts['gain_over_wrong_reference']=parts['wrong_reference']-parts['matched'];scores[f'{mode}_t{level}']=parts;values.append(parts['matched'])
            scores[mode+'_flow_mse']=sum(values)/len(values)
        score=(scores['reference_flow_mse']+scores['joint_flow_mse'])/2
        if score<best:best=score;save('best_flow.pt')
        for mode in ('reference','joint'):
            value=scores[mode+'_flow_mse']
            if value<best_modes.get(mode,float('inf')):best_modes[mode]=value;save(f'best_{mode}_flow.pt')
        save('latest.pt');gen_out=root/'generation_checks'/f'{step:06d}';gen_out.mkdir(parents=True,exist_ok=True)
        for mode,features in [('reference',image),('joint',joint)]:
            h=features[panel_ids].to(device)
            result=sample(model,h,torch.ones(h.shape[:2],device=device,dtype=torch.bool),seeds=text_seeds,steps=32,text_scale=2,embedding_adapter=True)
            for j,v in enumerate(result):Image.fromarray(decode(v)).save(gen_out/f'{panel[j]["case_id"]}_{mode}.png')
        render_checks(gen_out,panel,step)
        atomic_json(gen_out/'validation.json',scores)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(scores)+'\n')
        print('VALIDATION',json.dumps(scores),flush=True)
    if step==0:save('initial_adapter.pt');validate()
    end=min(cfg.train.steps,stop_after) if stop_after else cfg.train.steps
    print('TRAINING',step,end,flush=True)
    for update in range(step,end):
        if stopped[0]:break
        rng=torch.Generator().manual_seed(cfg.train.seed+1000003*update)
        ids=draw(groups['train'],cfg.train.batch_size,rng);choose=torch.rand(len(ids),generator=rng)<.5
        h=torch.where(choose[:,None,None],image[ids],joint[ids]).to(device)
        warm=min(1.,(update+1)/max(1,cfg.train.warmup));progress=max(0.,(update-cfg.train.warmup)/max(1,cfg.train.steps-cfg.train.warmup))
        lr=cfg.train.lr*warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss=objective(model,x[ids].to(device),h,device,cfg.train.seed+update+17)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite adapter loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(),1.,error_if_nonfinite=True);opt.step();step=update+1
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'loss':loss.item(),'lr':lr,'seconds':time.time()-start})+'\n')
        if step%cfg.train.eval_every==0 or step==end:validate()
    save('latest.pt')
    atomic_json(root/'status.json',{'step':step,'planned_steps':cfg.train.steps,'complete':step==cfg.train.steps,'stopped':stopped[0],
                'seconds':time.time()-start,'best_flow':best,'backbone_exact':backbone_digest(model)==baseline})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','probe','train']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--stop-after',type=int);a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.cuda.set_per_process_memory_fraction(.85,a.device)
    if a.stage=='smoke':smoke(a.out,a.device)
    elif a.stage=='probe':probe(a.out,a.device)
    else:train(a.out,a.device,a.stop_after)
