"""Held-out generation and reference sensitivity after lossless native16 training."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.curated_training import MixedData
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from train_visual_reference import thumbnail
from validate_resolution import category, noise_for

RUN = Path('outputs/reference_native16_lossless_20261107')
PRIOR = Path('outputs/reference_mixed_hq_20261107')
SEEDS = [2026110731, 2026110732]
VARIANTS = ('before16', 'prior32', 'after16')


def model_hash(weights):
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('native', 'inputs', 'panels'):
        (out/name).mkdir()
    data = MixedData(RUN)
    checkpoints = {}
    for name in ('best_joint_flow.pt', 'best_text_flow.pt', 'latest.pt'):
        path = RUN/'training'/name
        state = load_checkpoint(path)
        checkpoints[name] = {'path':str(path.resolve()),'step':state['step'],
                             'model_sha256':model_hash(state['model']),'file_sha256':file_hash(path)}
        del state
        gc.collect()
    checkpoint_info = {'checkpoints':checkpoints,
        'all_three_model_parameters_identical':len({c['model_sha256'] for c in checkpoints.values()})==1}
    atomic_json(out/'checkpoint_comparison.json',checkpoint_info)
    excluded = {r['target_hash'] for r in json.loads((RUN/'generation_checks/000270/panel.json').read_text())}
    excluded.update(r['target_hash'] for r in json.loads((RUN/'generation_checks/000270/fixed_text_different_references.json').read_text()))
    rng = random.Random(20261107311)
    selected = []
    quotas = [('train','highest_quality',12),('train','reviewed',12),
              ('val','highest_quality',12),('val','evaluation_only',12),('test','highest_quality',16)]
    for split,quality,count in quotas:
        groups = defaultdict(list)
        unique = defaultdict(list)
        for i,row in enumerate(data.rows):
            if (row['split'],row['quality_tier'])==(split,quality) and row['target_hash'] not in excluded:
                unique[row['target_hash']].append(i)
        for indices in unique.values():
            i = rng.choice(indices)
            groups[category(data.rows[i]['captions'][0])].append(i)
        for indices in groups.values():
            rng.shuffle(indices)
        chosen = []
        while len(chosen)<count:
            for key in sorted(groups):
                if groups[key] and len(chosen)<count:
                    chosen.append(groups[key].pop())
            if not any(groups.values()) and len(chosen)<count:
                raise ValueError('Insufficient fresh distinct targets')
        for i in chosen:
            row=data.rows[i]
            excluded.add(row['target_hash'])
            ident=f'{split}_{sum(r["split"]==split for r in selected):02d}'
            selected.append({**row,'case_id':ident,'pair_index':i})
    diversity=[]
    for prompt in ('helmet','sword','axe','shield'):
        pattern=r'\b(?:axe|battleaxe|greataxe|waraxe|handaxe)\b' if prompt=='axe' else r'\b'+prompt+r'\b'
        pools={'train':[],'holdout':[]}
        seen=set()
        for i,row in enumerate(data.rows):
            if row['quality_tier']!='highest_quality' or row['target_hash'] in seen:
                continue
            if re.search(pattern,row['captions'][0],re.I):
                pools['train' if row['split']=='train' else 'holdout'].append(i)
                seen.add(row['target_hash'])
        for pool in pools.values():
            rng.shuffle(pool)
        chosen=pools['train'][:2]+pools['holdout'][:2]
        chosen+= [i for i in pools['train'][2:]+pools['holdout'][2:] if i not in chosen][:4-len(chosen)]
        if len(chosen)!=4:
            raise ValueError('Need four distinct reference designs for '+prompt)
        for j,i in enumerate(chosen):
            diversity.append({**data.rows[i],'case_id':f'fixed_{prompt}_{j}',
                              'pair_index':i,'fixed_prompt':prompt})
    all_rows=selected+diversity
    for row in all_rows:
        ident=row['case_id']
        with Image.open(row['reference_file']) as im:
            thumbnail(im,256).save(out/'inputs'/f'{ident}_reference.png')
        with Image.open(row['target_file']) as im:
            assert im.size==(16,16)
            im.convert('RGBA').save(out/'inputs'/f'{ident}_target.png')
    atomic_json(out/'samples.json',selected)
    atomic_json(out/'diversity.json',diversity)
    indices=[r['pair_index'] for r in all_rows]
    fixed_prompts=[r['fixed_prompt'] for r in diversity]
    fixed_h,_=data.encoder.encode(conditioned_prompts(fixed_prompts,data.cfg),'cpu')
    common=['diamond sword','iron pickaxe','gold ingot','red gem','diamond helmet','iron axe','emerald','shield']
    base_text=DiskTextEncoder('outputs/qwen_vl_embedding_large_20261103/features',expected_identity=data.encoder.identity)
    common=[p for p in common if conditioned_prompts([p],data.cfg)[0] in base_text.index]
    common_h,_=base_text.encode(conditioned_prompts(common,data.cfg),'cpu')
    torch.save({'short':data.text[indices,0],'detail':data.text[indices,1],
        'reference':torch.from_numpy(np.array(data.features[indices],copy=True)).view(torch.bfloat16),
        'fixed_text':fixed_h,'common_text':common_h,'common_prompts':common,
        'target16':data.targets[indices],'text_identity':data.encoder.identity,
        'visual_identity':data.visual_identity},out/'features.pt')
    atomic_json(out/'protocol.json',{'source_run':str(RUN.resolve()),'source_pairs_sha256':data.pairs_sha256,
        'counts':dict(Counter(r['split'] for r in selected)),'distinct_primary_targets':len({r['target_hash'] for r in selected}),
        'selection_seed':20261107311,'generation_seeds':SEEDS,'steps':32,'solver':'heun',
        'text_scale':2.,'reference_scale':2.,'precision':'float32 for all model forwards',
        'selection':'fresh category-stratified targets, excludes this run monitoring examples; no output-based filtering',
        'modes':['text','reference','joint'],'primary_caption':'short filename label for every variant',
        'extra_caption':'after16 only, HQ detail text and joint; separate from primary comparisons',
        'variants':{'before16':'original approved native16 checkpoint at116 steps',
                    'prior32':'previous mixed native32 best checkpoint; contextual comparison, dataset differs',
                    'after16':'lossless native16 best_joint checkpoint at270 steps'},
        'noise':'CPU iid Gaussian32;16=2*average_pool2. Shared seeds and spatial randomness across variants.',
        'checkpoint_comparison':checkpoint_info,'features_sha256':file_hash(out/'features.pt'),
        'test_usage':'final evaluation only, never used for checkpoint selection',
        'holdout_caveat':'Unseen in this fine-tuning run; do not infer every MC target was unseen in base pretraining.',
        'reference_sensitivity':'four categories, four distinct designs each, fixed category text and seeds',
        'new_training':False})
    print('PREPARED',len(selected),'primary +',len(diversity),'fixed-text references',flush=True)


def load_model(variant,out,device):
    initialization=json.loads((RUN/'initialization.json').read_text())
    path=(Path(initialization['checkpoint']) if variant=='before16' else
          PRIOR/'training/best_joint_flow.pt' if variant=='prior32' else RUN/'training/best_joint_flow.pt')
    state=load_checkpoint(path)
    features=torch.load(out/'features.pt',weights_only=True)
    if state['text_identity']!=features['text_identity'] or state['visual_identity']!=features['visual_identity']:
        raise ValueError('Encoder identity mismatch')
    cfg=Experiment.from_dict(state['config'])
    model=PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    atomic_json(out/f'model_{variant}.json',{'checkpoint':str(path.resolve()),'step':state['step'],
        'native_size':cfg.model.size,'model_sha256':model_hash(state['model']),
        'checkpoint_sha256':file_hash(path)})
    del state,features
    gc.collect()
    return model


@torch.inference_mode()
def generate(out,variant,device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.30,device)
    features=torch.load(out/'features.pt',weights_only=True)
    rows=json.loads((out/'samples.json').read_text())
    diverse=json.loads((out/'diversity.json').read_text())
    model=load_model(variant,out,device)
    started=time.monotonic()
    done=0
    def batch_generate(ids,text_key,mode,names,seed,fixed=False):
        nonlocal done
        missing=[(i,name) for i,name in zip(ids,names) if not (out/'native'/name).exists()]
        if not missing:
            return
        chosen=[i for i,_ in missing]
        h=features[text_key][chosen].to(device)
        mask=torch.full(h.shape[:2],mode!='reference',dtype=torch.bool,device=device)
        refs=None if mode=='text' else features['reference'][[i+len(rows) if fixed else i for i in chosen]].to(device)
        result=sample(model,h,mask,refs,seeds=[seed]*len(chosen),steps=32,
                      initial_noise=noise_for([seed]*len(chosen),model.cfg.size,device))
        for value,(_,name) in zip(result,missing):
            Image.fromarray(decode(value)).save(out/'native'/name)
        done+=len(missing)
        atomic_json(out/f'progress_{variant}.json',{'phase':'running','mode':mode,'done':done,
                                                    'seconds':time.monotonic()-started})
    for si,seed in enumerate(SEEDS):
        for mode in ('text','reference','joint'):
            for start in range(0,len(rows),12):
                ids=list(range(start,min(start+12,len(rows))))
                batch_generate(ids,'short',mode,[f'{rows[i]["case_id"]}_{variant}_{mode}_s{si}.png' for i in ids],seed)
        for start in range(0,len(diverse),12):
            ids=list(range(start,min(start+12,len(diverse))))
            batch_generate(ids,'fixed_text','joint',[f'{diverse[i]["case_id"]}_{variant}_joint_s{si}.png' for i in ids],seed,fixed=True)
        for j in range(0,len(diverse),4):
            batch_generate([j],'fixed_text','text',[f'fixedtext_{diverse[j]["fixed_prompt"]}_{variant}_s{si}.png'],seed)
        ids=list(range(len(features['common_prompts'])))
        batch_generate(ids,'common_text','text',[f'common_{i:02d}_{variant}_s{si}.png' for i in ids],seed)
        if variant=='after16':
            ids=[i for i,r in enumerate(rows) if r['quality_tier']=='highest_quality']
            for mode in ('text','joint'):
                for start in range(0,len(ids),12):
                    group=ids[start:start+12]
                    batch_generate(group,'detail',mode,[f'{rows[i]["case_id"]}_{variant}_{mode}_detail_s{si}.png' for i in group],seed)
    # Fixed x_t and captions: compare correct, permuted and absent references.
    flow=[]
    count=len(rows)
    target=features['target16'][:count].to(device)
    if model.cfg.size==32:
        target=target.repeat_interleave(2,2).repeat_interleave(2,3)
    h=features['short'][:count].to(device)
    mask=torch.ones(h.shape[:2],dtype=torch.bool,device=device)
    reference=features['reference'][:count].to(device)
    for mode in ('joint','reference'):
        current_mask=mask if mode=='joint' else torch.zeros_like(mask)
        for level in (.1,.5,.9):
            noise=noise_for([SEEDS[0]]*count,model.cfg.size,device)
            x=target*(1-level)+noise*level
            scores={}
            for label,refs in [('matched',reference),('wrong_reference_same_text',reference.roll(1,0)),('no_reference',None)]:
                parts=[]
                for start in range(0,count,12):
                    stop=min(start+12,count)
                    ctx=model.context(h[start:stop],current_mask[start:stop],None if refs is None else refs[start:stop],None)
                    prediction=model(x[start:stop],torch.full((stop-start,),level,device=device),ctx)
                    parts.extend((prediction-(noise-target)[start:stop]).square().mean((1,2,3)).cpu().tolist())
                scores[label]=parts
            flow.append({'mode':mode,'level':level,'scores':scores})
    atomic_json(out/f'flow_{variant}.json',flow)
    atomic_json(out/f'progress_{variant}.json',{'phase':'complete','outputs_written':done,
                                               'seconds':time.monotonic()-started})
    print('COMPLETE',variant,done,round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','generate'])
    parser.add_argument('--out',type=Path,default=Path('outputs/native16_lossless_eval_20261107'))
    parser.add_argument('--variant',choices=VARIANTS,default='after16')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare(args.out)
    else:
        generate(args.out,args.variant,args.device)
