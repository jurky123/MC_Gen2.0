"""Explicit hue-counterfactual pilot: materialize, train, evaluate, no test tuning."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

from mcgen.catalog import read_catalog, write_catalog, atomic_json, file_hash
from mcgen.color_supervision import shift_hue, color_summary, hue_distance, compatible_color_pair, category_from_caption as family
from mcgen.config import Experiment
from mcgen.dataset import CatalogDataset
from mcgen.evaluation import metrics
from mcgen.images import native_grid, content_hash
from mcgen.model import PixelFlow
from mcgen.runtime import Runtime, load_checkpoint
from mcgen.training import Trainer
from run_attribute_pilot import checker, FAMILIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/color_counterfactual_focused_20261102'
PARENT = ROOT / 'outputs/attribute_prior_20261102'
BASE = PARENT / 'adapter/best_flow.pt'
CONFIG = ROOT / 'configs/v3/experiments/color_counterfactual.yaml'
TRAIN_ANGLES = (0, 60, 120, 180, 240, 300)
DEV_ANGLES = (30, 90, 150, 210, 270, 330)
SUPPORTED = {'pickaxe','sword','axe','shovel','helmet','gem','door','ingot'}


def prepare():
    records, meta = read_catalog(ROOT/'data/catalogs/native16_v3/catalog')
    parent_records, parent_meta = read_catalog(PARENT/'catalog')
    exposed = {r['target_hash'] for r in parent_records if r['split']=='train'}
    data = CatalogDataset(records,16,128,'train',asset_type='item',target_scope='paired')
    unique = {}
    for r in sorted(records,key=lambda r:r['id']):
        if r['split'] in ('train','val') and r['asset_type']=='item' and r.get('reference') and family(r['prompt'])!='other':
            unique.setdefault(r['target_hash'],r)
    groups=defaultdict(list); images={}; rejects=Counter()
    for r in unique.values():
        target=native_grid(data.array(r['target']),16);ref=data.array(r['reference'])
        keep,details=compatible_color_pair(target,ref)
        if not keep:rejects['weak_or_mismatched_color']+=1;continue
        f=family(r['prompt']);groups[r['split'],f].append(r);images[r['id']]=(target,ref)
    selected={'train':[],'val':[],'dev':[]};rng=random.Random(20261103)
    for f in FAMILIES:
        train=groups['train',f];val=groups['val',f];rng.shuffle(train);rng.shuffle(val)
        # Small ambiguous families lack held-out coverage; do not silently
        # train cake/jar/compound-object labels as canonical object shapes.
        if f not in SUPPORTED:
            continue
        selected['train']+=train[:250]
        n=min(4,len(val)//3)
        selected['dev']+=val[:n];selected['val']+=val[n:n+8]
    if len(selected['dev'])<12 or len(selected['val'])<12:raise ValueError('insufficient matched-color validation families')
    # Audit transformed hashes before materialization, removing whole source groups.
    owners=defaultdict(set);hashes={};bad=set()
    for split,rows in selected.items():
        for r in rows:
            angles=DEV_ANGLES if split=='dev' else TRAIN_ANGLES
            hashes[r['id']]=[content_hash(shift_hue(images[r['id']][0],a)) for a in angles]
            for h in hashes[r['id']]:
                owners[h].add(split)
                if split!='train' and h in exposed:bad.add(r['id'])
    collision={h for h,roles in owners.items() if len(roles)>1}
    for rows in selected.values():
        for r in rows:
            if any(h in collision for h in hashes[r['id']]):bad.add(r['id'])
    selected={s:[r for r in rows if r['id'] not in bad] for s,rows in selected.items()}
    if not selected['dev'] or not selected['val']:raise ValueError('all held-out variants collided')
    if (OUT/'target.mmap').exists():raise ValueError('prepared outputs already exist')
    n=sum(len(rows)*6 for rows in selected.values())
    targets=np.memmap(OUT/'target.mmap',mode='w+',dtype=np.uint8,shape=(n,16,16,4))
    refs=np.memmap(OUT/'reference.mmap',mode='w+',dtype=np.uint8,shape=(n,128,128,4))
    catalog=[];dev=[];j=0;coverage={}
    for split,rows in selected.items():
        coverage[split]=dict(Counter(family(r['prompt']) for r in rows))
        for r in rows:
            f=family(r['prompt']);target,ref=images[r['id']]
            for angle in DEV_ANGLES if split=='dev' else TRAIN_ANGLES:
                a,b=shift_hue(target,angle),shift_hue(ref,angle)
                assert np.array_equal(a[...,3],target[...,3]) and np.array_equal(b[...,3],ref[...,3])
                assert b.shape==(128,128,4)
                targets[j]=a;refs[j]=b
                row={**r,'id':f"color-cf:{r['id']}:{angle}",'prompt':f'a {f} inventory item',
                     'target_hash':content_hash(a),'source_target_hash':r['target_hash'],'source_id':r['id'],
                     'source_prompt':r['prompt'],'family':f,'hue_shift_degrees':angle,
                     'target':{'path':str(OUT/'target.mmap'),'shape':[n,16,16,4],'index':j},
                     'reference':{'path':str(OUT/'reference.mmap'),'shape':[n,128,128,4],'index':j}}
                (dev if split=='dev' else catalog).append(row);j+=1
    targets.flush();refs.flush()
    out_meta=write_catalog(OUT/'catalog',catalog,{'native_size':16,'parent_fingerprint':meta['fingerprint'],
        'transform':'HSV hue rotation; alpha, max/min RGB and native grid preserved',
        'caption':'category-only weak labels, no color/material words', 'quarantined_source_groups':sorted(bad)})
    (OUT/'dev.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dev))
    atomic_json(OUT/'protocol.json',{'initial_checkpoint':str(BASE),'initial_sha256':file_hash(BASE),
        'config_sha256':file_hash(CONFIG),'catalog_fingerprint':out_meta['fingerprint'],'catalog_audit':out_meta['audit'],
        'source_family_counts':coverage,'rejected':dict(rejects),'quarantined_source_groups':sorted(bad),
        'supported_families':sorted(SUPPORTED),
        'dev_sha256':file_hash(OUT/'dev.jsonl'),'train_angles':TRAIN_ANGLES,'dev_angles':DEV_ANGLES,
        'checkpoint_selection':'minimum joint_flow_mse; 2000 fixed steps; no development-generation tuning',
        'generation':{'seeds':[20261103,20261104],'steps':40,'solver':'heun','text_scale':2.,'ref_scale':2.},
        'limits':['Development validation projects, not untouched final test','Weak category labels; preview inspection is not full annotation',
            'Synthetic hue edits, not material transfer','Held-out shift offsets are not proof of unseen absolute colors',
            'One training seed; frozen backbone; no vanilla-style guarantee']})
    preview=[]
    for f in FAMILIES:preview += [r for r in selected['train'] if family(r['prompt'])==f][:2]
    sheet=Image.new('RGB',(8*104,((len(preview)+3)//4)*124+24),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(preview):
        x,y=(i%4)*208,(i//4)*124;draw.text((x,y),family(r['prompt'])+':'+r['id'],fill='black')
        a,b=images[r['id']];sheet.paste(checker(a),(x,y+20));sheet.paste(checker(b),(x+104,y+20))
    sheet.save(OUT/'source_preview.png')
    print(json.dumps({'audit':out_meta['audit'],'families':coverage,'dev_variants':len(dev),'quarantined':len(bad)}),flush=True)


def train(device):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert file_hash(CONFIG)==protocol['config_sha256'] and file_hash(BASE)==protocol['initial_sha256']
    trainer=Trainer(Experiment.load(CONFIG),OUT/'catalog',OUT/'train',device,device,initialize=BASE)
    trainer.run()
    before=load_checkpoint(BASE)['model'];after=load_checkpoint(OUT/'train/best_joint_flow.pt')['model']
    inherited=[k for k in before if not Trainer.is_attribute(k)]
    assert all(torch.equal(before[k],after[k]) for k in inherited)
    atomic_json(OUT/'backbone_verification.json',{'frozen_tensors':len(inherited),'bitwise_identical':True})


def color_metrics(output,target):
    a,b=color_summary(output),color_summary(target);error=hue_distance(a['hue_degrees'],b['hue_degrees'])
    valid=a['chromatic_pixels']>=8 and a['concentration']>=.4
    # Missing/ambiguous color is failure, not a free exclusion from averages.
    return {'hue_error_degrees':error if valid and error is not None else 180.,
            'color_valid':valid,'chroma_coverage':a['chromatic_coverage'],'mean_chroma':a['mean_chroma'],
            'hue_within_30':bool(valid and error is not None and error<=30)}


def evaluate(device):
    protocol=json.loads((OUT/'protocol.json').read_text());assert file_hash(OUT/'dev.jsonl')==protocol['dev_sha256']
    rows=[json.loads(x) for x in (OUT/'dev.jsonl').read_text().splitlines()]
    ds=CatalogDataset(rows,16,128,'val',asset_type='item');out=OUT/'evaluation';out.mkdir(exist_ok=False);(out/'images').mkdir()
    runtime=Runtime(BASE,device,device);results=[];images={};selection={};regression=[]
    regression_cases=[json.loads(x) for x in (PARENT/'dev_paired.jsonl').read_text().splitlines()][:24]
    by_id={r['id']:i for i,r in enumerate(rows)}
    source_groups=defaultdict(dict)
    for i,r in enumerate(rows):source_groups[r['source_id']][r['hue_shift_degrees']]=i
    seeds=protocol['generation']['seeds']
    for name,checkpoint in [('previous',BASE),('counterfactual',OUT/'train/best_joint_flow.pt')]:
        state=load_checkpoint(checkpoint);runtime.config=Experiment.from_dict(state['config'])
        runtime.model=PixelFlow(runtime.config.model).to(device);runtime.model.load_state_dict(state['model']);runtime.model.eval()
        assert runtime.encoder.identity==state['text_identity']
        selection[name]={'checkpoint':str(checkpoint),'sha256':file_hash(checkpoint),'step':state['step']}
        for seed in seeds:
            for mode in ('text','correct','wrong_color','rotated'):
                for start in range(0,len(rows),32):
                    chunk=rows[start:start+32];refs=None
                    if mode!='text':
                        chosen=[rows[source_groups[r['source_id']][(r['hue_shift_degrees']+120)%360]] if mode=='wrong_color' else r for r in chunk]
                        refs=[ds.array(r['reference']) for r in chosen]
                        if mode=='rotated':refs=[np.rot90(a).copy() for a in refs]
                    # Same source and seed across hue variants, so color is the sole changed input.
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        generated=runtime.generate([r['prompt'] for r in chunk],refs,seeds=[seed]*len(chunk),steps=40,
                            text_scale=2.,ref_scale=0. if mode=='text' else 2.,reference_mode='attributes',solver='heun')
                    for i,(r,a) in enumerate(zip(chunk,generated)):
                        index=start+i;key=f'{name}_{seed}_{mode}_{index:04d}';path=f'images/{key}.png';Image.fromarray(a).save(out/path)
                        target=ds.array(r['target']);scores=metrics(a,target);scores.update(color_metrics(a,target))
                        if mode=='wrong_color':
                            swapped=rows[source_groups[r['source_id']][(r['hue_shift_degrees']+120)%360]]
                            scores['hue_error_to_supplied_color']=color_metrics(a,ds.array(swapped['target']))['hue_error_degrees']
                        images[name,seed,mode,index]=a
                        results.append({'model':name,'seed':seed,'mode':mode,'id':r['id'],'source_id':r['source_id'],
                            'lineage':r['lineage'],'family':r['family'],'angle':r['hue_shift_degrees'],'image':path,'metrics':scores})
                print('EVAL',name,seed,mode,len(rows),flush=True)
        # Keep the image-only and explicit shape entrypoints visible in the audit.
        for mode in ('reference','shape'):
            refs=[ds.array(r['reference']) for r in regression_cases]
            prompts=['' if mode=='reference' else r['prompt'] for r in regression_cases]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=runtime.generate(prompts,refs,seeds=[20261103+i for i in range(len(prompts))],steps=40,
                    text_scale=0. if mode=='reference' else 2.,ref_scale=2.,reference_mode='shape',solver='heun')
            for i,(r,a) in enumerate(zip(regression_cases,outputs)):
                path=f'images/{name}_regression_{mode}_{i:03d}.png';Image.fromarray(a).save(out/path)
                regression.append({'model':name,'mode':mode,'id':r['id'],'lineage':r['lineage'],'image':path,
                                   'metrics':metrics(a,native_grid(ds.array(r['target']),16))})
    summary={}
    for name in selection:
        for mode in ('text','correct','wrong_color','rotated'):
            selected=[r for r in results if r['model']==name and r['mode']==mode];summary[name+'/'+mode]={}
            for metric in ('composite_mae','alpha_iou','hue_error_degrees','hue_within_30','color_valid','mean_chroma'):
                projects=defaultdict(list)
                for r in selected:projects[r['lineage']].append(float(r['metrics'][metric]))
                summary[name+'/'+mode][metric]=float(np.mean([np.mean(v) for v in projects.values()]))
    rotation={};shape_stability={}
    for name in selection:
        rotation[name]=max(float(np.abs(images[name,s,'correct',i].astype(float)-images[name,s,'rotated',i].astype(float)).max()) for s in seeds for i in range(len(rows)))
        stable=[]
        for g in source_groups.values():
            indices=list(g.values())
            for s in seeds:
                for i in indices[1:]:stable.append(metrics(images[name,s,'correct',indices[0]],images[name,s,'correct',i])['alpha_iou'])
        shape_stability[name]=float(np.mean(stable))
    for seed in seeds:
        assert all(np.array_equal(images['previous',seed,'text',i],images['counterfactual',seed,'text',i]) for i in range(len(rows)))
    # One source per row, all six fixed hues; separate sheets retain every case.
    for name in selection:
        groups=list(source_groups.items())
        for page in range((len(groups)+7)//8):
            sheet=Image.new('RGB',(12*104,8*124+24),'white');draw=ImageDraw.Draw(sheet)
            for j,angle in enumerate(DEV_ANGLES):draw.text((j*208,2),f'{angle} deg: target / output',fill='black')
            for k,(source,g) in enumerate(groups[page*8:page*8+8]):
                first=rows[next(iter(g.values()))];draw.text((0,24+k*124),first['family']+' '+source,fill='black')
                for j,angle in enumerate(DEV_ANGLES):
                    i=g[angle];sheet.paste(checker(ds.array(rows[i]['target'])),(j*208,44+k*124))
                    sheet.paste(checker(images[name,seeds[0],'correct',i]),(j*208+104,44+k*124))
            sheet.save(out/f'{name}_colors_{page:02d}.png')
    (out/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    (out/'regression_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in regression))
    regression_summary={}
    for name in selection:
        for mode in ('reference','shape'):
            chosen=[r for r in regression if r['model']==name and r['mode']==mode]
            regression_summary[name+'/'+mode]={m:float(np.mean([r['metrics'][m] for r in chosen])) for m in ('composite_mae','alpha_iou')}
    atomic_json(out/'report.json',{'selection':selection,'project_mean_metrics':summary,'source_cases':len(source_groups),
        'variants':len(rows),'samples':len(results),'seeds':seeds,'rotation_max_absolute_pixel_change':rotation,
        'across_color_alpha_iou':shape_stability,'text_outputs_bitwise_preserved':True,
        'regression_sample_means_24_cases':regression_summary,'limits':protocol['limits']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','evaluate'));parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    if args.stage=='prepare':prepare()
    elif args.stage=='train':train(args.device)
    else:evaluate(args.device)
