"""Small matched-category 2x2 data-curation / learning-rate diagnostic."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from mcgen.catalog import atomic_json, file_hash, write_catalog
from mcgen.images import composite, content_hash
from mcgen.tasks import conditioned_prompts

ROOT=Path('outputs/pixel_quality_pilot_20261103')
SOURCE=Path('outputs/fullscale_text10k_20261103')
PARENT=SOURCE/'training/best_flow.pt'
CATS='pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot gem apple book bow'.split()
MAPS={}


def array(r):
    s=r['target'];key=(s['path'],tuple(s['shape']))
    if key not in MAPS:MAPS[key]=np.memmap(key[0],mode='r',dtype='uint8',shape=key[1])
    a=np.array(MAPS[key][s['index']]);assert content_hash(a)==r['target_hash'];return a


def prepare():
    ROOT.mkdir(exist_ok=False)
    old=json.load(open('outputs/text_expansion2k_20261102/cases.json'))
    known={r['target_hash']:r for r in old if r.get('caption_origin')=='expansion_automatic_weak_label'}
    clean=defaultdict(list);control=defaultdict(list);val=defaultdict(list)
    for line in (SOURCE/'catalog/samples.jsonl').open():
        r=json.loads(line)
        if r['caption_variant']!='short':continue
        c=r.get('category_15')
        if c not in CATS:continue
        r['category']=c
        if r['split']=='train':
            if r['target_hash'] in known:clean[c].append(r)
            else:control[c].append(r)
        elif r['split']=='val' and r['target_hash'] in known:val[c].append(r)
    rng=random.Random(202611031);groups={}
    for name,pools,n in [('clean_candidates',clean,12),('control',control,8),('validation_candidates',val,3)]:
        rows=[]
        for c in CATS:
            rng.shuffle(pools[c]);assert len(pools[c])>=n,(name,c,len(pools[c]));rows.extend(pools[c][:n])
        groups[name]=rows
        for page in range(0,len(rows),60):
            batch=rows[page:page+60];sheet=Image.new('RGB',(1600,40+160*((len(batch)+9)//10)),'white');d=ImageDraw.Draw(sheet)
            d.text((8,8),f'{name} indices {page}..{page+len(batch)-1}',fill='black')
            for j,r in enumerate(batch):
                x=j%10*160;y=40+j//10*160
                im=Image.fromarray(composite(array(r),128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST)
                sheet.paste(im,(x+32,y));d.multiline_text((x+3,y+100),'\n'.join(textwrap.wrap(f'{page+j}: {r["source_filename"]}',24)[:4]),fill='black')
            sheet.save(ROOT/f'{name}_{page//60}.png')
    atomic_json(ROOT/'candidate_groups.json',groups)
    atomic_json(ROOT/'plan.json',{'parent':str(PARENT),'parent_sha256':file_hash(PARENT),'train_per_arm':120,
        'categories':CATS,'per_category':8,'additional_updates':300,'effective_batch':64,
        'learning_rates':[1e-5,2e-6],'optimizer_and_global_step_preserved':True,
        'caption_policy':'Filename-only in all arms to isolate image selection from automatic-description coverage.',
        'data_control':'Class-matched random candidates from existing fullscale training corpus, not synthetic corruption. Clean arm is separately visually screened. Different images/projects remain a confound; this tests a curation package, not isolated contamination causality.',
        'scope':'Development diagnostic; not a fullscale training restart or automatic model promotion.'})
    print({k:len(v) for k,v in groups.items()},flush=True)


def build():
    groups=json.load(open(ROOT/'candidate_groups.json'));review=json.load(open(ROOT/'visual_review.json'))
    candidates=groups['clean_candidates'];reject=set(review['clean_excluded_indices']);clean=[]
    for c in CATS:
        pool=[r for i,r in enumerate(candidates) if r['category']==c and i not in reject]
        assert len(pool)>=8,(c,len(pool));clean.extend(pool[:8])
    val=[r for i,r in enumerate(groups['validation_candidates']) if i not in review['validation_excluded_indices']]
    assert len(val)>=30
    for name,train in [('clean',clean),('control',groups['control'])]:
        write_catalog(ROOT/name/'catalog',train+val,{'native_size':16,'purpose':'matched-category pixel quality pilot',
            'review_sha256':file_hash(ROOT/'visual_review.json'),'parent_catalog':str(SOURCE/'catalog'),
            'caption_policy':'filename_only','test_used':False})
        atomic_json(ROOT/name/'train_cases.json',train)
        for lrname,lr in [('normal',1e-5),('low',2e-6)]:
            cfg=yaml.safe_load((SOURCE/'config.yaml').read_text());cfg['train'].update(lr=lr,eval_every=100,val_samples=len(val))
            path=ROOT/(name+'_'+lrname);path.mkdir(exist_ok=True);(path/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    evaluation=[]
    for r in json.load(open('outputs/fullscale_eval6000_20261103/eval_cases.json'))['cases']:
        if r['group']=='novel_composition':evaluation.append(r)
    for r in val:evaluation.append({**r,'evaluation_prompt':r['prompt'],'group':'clean_val','seed':202611032})
    for name,train in [('clean',clean),('control',groups['control'])]:
        for c in CATS:
            r=next(r for r in train if r['category']==c)
            evaluation.append({**r,'evaluation_prompt':r['prompt'],'group':name+'_fit','seed':202611032})
    broad=[r for r in json.load(open('outputs/fullscale_eval6000_20261103/eval_cases.json'))['cases'] if r['group']=='fullscale_val_filename']
    evaluation.extend({**r,'group':'broad_val'} for r in broad[:32])
    atomic_json(ROOT/'eval_cases.json',evaluation);print('built',len(evaluation),'evaluation cases',flush=True)


def cache(device):
    from mcgen.config import Experiment
    from mcgen.conditioning import TextEncoder
    from mcgen.disk_text import DiskTextEncoder
    cfg=Experiment.load(ROOT/'clean_normal/config.yaml');rows=[]
    for name in ['clean','control']:
        rows.extend(json.loads(s) for s in (ROOT/name/'catalog/samples.jsonl').open())
    cases=json.load(open(ROOT/'eval_cases.json'))
    prompts=sorted(set(conditioned_prompts([r['prompt'] for r in rows]+[r['evaluation_prompt'] for r in cases]+[''],cfg)))
    source=DiskTextEncoder(SOURCE/'text_features');missing=[p for p in prompts if p not in source.index]
    encoder=None
    if missing:
        encoder=TextEncoder(cfg.text,cfg.model,device);assert encoder.identity==source.identity
    out=ROOT/'features';out.mkdir(exist_ok=False);shape=(len(prompts),128,12288)
    f=np.memmap(out/'features.uint16.mmap',mode='w+',dtype='uint16',shape=shape)
    m=np.memmap(out/'masks.uint8.mmap',mode='w+',dtype='uint8',shape=shape[:2])
    for i in range(0,len(prompts),16):
        batch=prompts[i:i+16]
        for j,p in enumerate(batch):
            h,mask=(source if p in source.index else encoder).encode([p],'cpu')
            f[i+j]=h[0].view(torch.uint16).numpy();m[i+j]=mask[0].numpy()
        print('cached',min(i+16,len(prompts)),len(prompts),flush=True)
    f.flush();m.flush();atomic_json(out/'prompts.json',prompts)
    atomic_json(out/'metadata.json',{'shape':list(shape),'complete':True,'text_identity':source.identity,
        'prompts_sha256':file_hash(out/'prompts.json'),'missing_computed_online':len(missing)})


def run(data,device):
    from mcgen.config import Experiment
    from mcgen.disk_text import DiskTextEncoder
    from mcgen.training import Trainer
    from mcgen.runtime import load_checkpoint
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,device)
    encoder=DiskTextEncoder(ROOT/'features')
    for rate in ['normal','low']:
        name=data+'_'+rate;out=ROOT/name;cfg=Experiment.load(out/'config.yaml')
        state=load_checkpoint(PARENT);assert state['step']==2000
        trainer=Trainer(cfg,ROOT/data/'catalog',out/'training',device,device,initialize=PARENT,encoder=encoder)
        trainer.optimizer.load_state_dict(state['optimizer']);trainer.step=state['step'];del state
        atomic_json(out/'baseline_validation.json',trainer.validate())
        atomic_json(out/'status.json',{'stage':'training','start_step':2000,'end_step':2300})
        begin=time.time();trainer.run(stop_after=2300)
        assert trainer.step==2300
        atomic_json(out/'status.json',{'stage':'complete','step':trainer.step,'seconds':time.time()-begin})
        del trainer;torch.cuda.empty_cache()
    print('complete',data,flush=True)


def evaluate(device):
    from mcgen.config import Experiment
    from mcgen.disk_text import DiskTextEncoder
    from mcgen.runtime import load_checkpoint
    from mcgen.model import PixelFlow
    from mcgen.sampling import sample
    from mcgen.images import decode
    from mcgen.evaluation import metrics
    from diagnose_pixel_integrity import stats
    from evaluate_text_recovery import color_fraction
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,device)
    cfg=Experiment.load(ROOT/'clean_normal/config.yaml');encoder=DiskTextEncoder(ROOT/'features');cases=json.load(open(ROOT/'eval_cases.json'))
    arms={'before':PARENT,**{n:ROOT/n/'training/latest.pt' for n in ['clean_normal','clean_low','control_normal','control_low']}}
    plan=json.load(open(ROOT/'plan.json'));assert file_hash(PARENT)==plan['parent_sha256']
    for arm,path in arms.items():
        assert path.is_file(),path
        if arm!='before':assert json.load(open(ROOT/arm/'status.json'))['stage']=='complete',arm
    model=PixelFlow(cfg.model).to(device);out=ROOT/'evaluation';out.mkdir(exist_ok=False);results=[]
    atomic_json(out/'protocol.json',{'case_sha256':file_hash(ROOT/'eval_cases.json'),'steps':32,'solver':'heun','text_scale':2,'precision':'float32','endpoints':'All four final step2300 checkpoints, not validation-selected bests','checkpoint_sha256':{arm:file_hash(path) for arm,path in arms.items()}})
    for arm,path in arms.items():
        state=load_checkpoint(path);assert state['step']==(2000 if arm=='before' else 2300)
        model.load_state_dict(state['model']);del state;model.eval();(out/arm).mkdir()
        for start in range(0,len(cases),8):
            chunk=cases[start:start+8];text,mask=encoder.encode(conditioned_prompts([r['evaluation_prompt'] for r in chunk],cfg),device)
            with torch.inference_mode():raw=sample(model,text,mask,seeds=[r['seed'] for r in chunk],steps=32,text_scale=2,solver='heun')
            for j,(r,v) in enumerate(zip(chunk,raw)):
                a=decode(v);Image.fromarray(a).save(out/arm/f'{start+j:03d}.png')
                score=metrics(a,array(r)) if r.get('target') else {}
                if r.get('requested_color'):score['requested_color_fraction']=color_fraction(a,r['requested_color'])
                results.append({'arm':arm,'group':r['group'],'case':start+j,'metrics':score,'pixel_stats':stats(a)})
        print('evaluated',arm,flush=True)
    atomic_json(out/'results.json',results)
    summary={}
    for group in dict.fromkeys(r['group'] for r in cases):
        summary[group]={}
        for arm in arms:
            rows=[r for r in results if r['group']==group and r['arm']==arm];values={}
            for field in ['metrics','pixel_stats']:
                values[field]={k:float(np.mean([r[field][k] for r in rows if r[field][k] is not None])) for k in rows[0][field] if any(r[field][k] is not None for r in rows)}
            summary[group][arm]={'n':len(rows),**values}
        indices=[i for i,r in enumerate(cases) if r['group']==group]
        for page in range(0,min(len(indices),36),12):
            batch=indices[page:page+12];has_target=bool(cases[batch[0]].get('target'));offset=166 if has_target else 0
            sheet=Image.new('RGB',(1180+offset,40+len(batch)*110),'white');d=ImageDraw.Draw(sheet)
            if has_target:d.text((330,8),'target',fill='black')
            for col,arm in enumerate(arms):d.text((330+offset+col*166,8),arm,fill='black')
            for row,i in enumerate(batch):
                y=40+row*110;d.multiline_text((5,y),'\n'.join(textwrap.wrap(cases[i]['evaluation_prompt'],43)[:6]),fill='black')
                if has_target:sheet.paste(Image.fromarray(composite(array(cases[i]),128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),(330,y))
                for col,arm in enumerate(arms):
                    a=np.array(Image.open(out/arm/f'{i:03d}.png'))
                    sheet.paste(Image.fromarray(composite(a,128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),(330+offset+col*166,y))
            sheet.save(out/f'{group}_{page//12}.png')
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','build','cache','run','evaluate']);p.add_argument('--data',choices=['clean','control']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage in ('prepare','build'):globals()[a.stage]()
    elif a.stage=='run':run(a.data,a.device)
    else:globals()[a.stage](a.device)
