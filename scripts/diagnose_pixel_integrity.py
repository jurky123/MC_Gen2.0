"""Fixed-input diagnostics: source pixels, cache identity, precision and sampling."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import random
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.images import rgba, encode, decode, composite, content_hash, native_grid

ROOT=Path('outputs/pixel_integrity_diagnosis_20261103')
RUN=Path('outputs/fullscale_text10k_20261103')
EVAL=Path('outputs/fullscale_eval6000_20261103')


def components(mask):
    pending=set(zip(*np.where(mask)));sizes=[]
    while pending:
        start=pending.pop();stack=[start];n=0
        while stack:
            y,x=stack.pop();n+=1
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    q=(y+dy,x+dx)
                    if q in pending:pending.remove(q);stack.append(q)
        sizes.append(n)
    return sorted(sizes,reverse=True)


def stats(image):
    a=rgba(image);alpha=a[...,3];mask=alpha>=128;visible=alpha>8
    sizes=components(mask);pixels=int(mask.sum())
    colors=a[...,:3][mask]
    return {'visible_pixels':int(visible.sum()),'foreground_pixels':pixels,
        'fractional_alpha_among_visible':float(((alpha>8)&(alpha<247)).sum()/max(1,visible.sum())),
        'components8':len(sizes),'tiny_island_pixels':sum(n for n in sizes[1:] if n<=2),
        'largest_component_fraction':max(sizes,default=0)/max(1,pixels),
        'rgb_colors_foreground':len(np.unique(colors,axis=0)),
        'coarse_rgb_bins_foreground':len(np.unique(colors//32,axis=0))}


def means(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def data():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2)
    rng=random.Random(20261103);sample=[];n=0;counts=Counter()
    for line in (RUN/'catalog/samples.jsonl').open():
        r=json.loads(line)
        if r['split']!='train' or r['caption_variant']!='short':continue
        n+=1;counts[r['alpha_mode']]+=1
        if len(sample)<2048:sample.append(r)
        else:
            j=rng.randrange(n)
            if j<len(sample):sample[j]=r
    maps={};results=[];source_equal=0;roundtrip_max=0;arrays=[]
    def load(spec):
        key=(spec['path'],tuple(spec['shape']))
        if key not in maps:maps[key]=np.memmap(key[0],mode='r',dtype='uint8',shape=key[1])
        return np.array(maps[key][spec['index']])
    for i,r in enumerate(sample):
        a=native_grid(load(r['target']),16);assert content_hash(a)==r['target_hash']
        error=int(np.abs(decode(encode(a)).astype(int)-a.astype(int)).max());roundtrip_max=max(roundtrip_max,error)
        if i<128:
            original=native_grid(load(r['source_target']),16);assert np.array_equal(a,original);source_equal+=1
        results.append({'id':r['id'],'filename':r['source_filename'],**stats(a)})
        if i<64:arrays.append(a)
    atomic_json(ROOT/'data_audit.json',{'train_targets':n,'alpha_modes':dict(counts),'sample_size':len(sample),
        'source_pixels_equal_count':source_equal,'encode_decode_max_uint8_error':roundtrip_max,'mean':means([{k:v for k,v in r.items() if k not in ('id','filename')} for r in results]),'sample_records':results})
    sheet=Image.new('RGB',(1280,1280),'white');draw=ImageDraw.Draw(sheet)
    for i,a in enumerate(arrays):
        x=i%8*160;y=i//8*160
        sheet.paste(Image.fromarray(composite(a,128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),(x+32,y+6))
        draw.multiline_text((x+4,y+108),'\n'.join(textwrap.wrap(results[i]['filename'],23)[:3]),fill='black')
    sheet.save(ROOT/'train_sample64.png')
    generated={}
    for label,folder,arm in [('before','evaluation','before'),('best2000','evaluation_best_flow','recovery'),('latest6000','evaluation','recovery')]:
        cases=json.load(open(EVAL/'eval_cases.json'))['cases'];rows=[]
        for i,c in enumerate(cases):
            rows.append({'group':c['group'],**stats(np.array(Image.open(EVAL/folder/'images'/f'{arm}_{i:03d}.png').convert('RGBA')))})
        generated[label]={g:means([{k:v for k,v in r.items() if k!='group'} for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    atomic_json(ROOT/'existing_generation_stats.json',generated)
    print(json.dumps({'stage':'data_complete','roundtrip_max_error':roundtrip_max,'source_equal':source_equal,'sample_mean':means([{k:v for k,v in r.items() if k not in ('id','filename')} for r in results])}),flush=True)


def gpu(device):
    from mcgen.config import Experiment
    from mcgen.conditioning import TextEncoder
    from mcgen.disk_text import DiskTextEncoder
    from mcgen.model import PixelFlow
    from mcgen.runtime import load_checkpoint
    from mcgen.sampling import sample
    from mcgen.tasks import conditioned_prompts
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.40,device)
    cfg=Experiment.load(RUN/'config.yaml');print('loading_qwen',flush=True)
    encoder=TextEncoder(cfg.text,cfg.model,device);cache=DiskTextEncoder(RUN/'text_features',encoder.identity)
    all_prompts=json.load(open(RUN/'text_features/prompts.json'));cache_checks=[]
    for start in [0,(len(all_prompts)//2)//16*16,len(all_prompts)//16*16-16]:
        ps=all_prompts[start:start+16];fresh,mask=encoder.encode(ps,'cpu');stored,sm=cache.encode(ps,'cpu')
        valid=(fresh.float()-stored.float())[mask]
        cache_checks.append({'start':start,'mask_equal':torch.equal(mask,sm),'all_bitwise_equal':torch.equal(fresh,stored),
            'valid_max_abs':valid.abs().max().item(),'valid_mean_abs':valid.abs().mean().item(),
            'valid_relative_l2':(valid.norm()/fresh.float()[mask].norm()).item()})
    atomic_json(ROOT/'cache_checks.json',cache_checks);print('cache_checks',cache_checks,flush=True)
    cases=[r for r in json.load(open(EVAL/'eval_cases.json'))['cases'] if r['group']=='novel_composition' and r['seed']==20261112]
    assert len(cases)==18
    text,mask=encoder.encode(conditioned_prompts([r['evaluation_prompt'] for r in cases],cfg),device)
    seeds=[r['seed'] for r in cases];model=PixelFlow(cfg.model).to(device)
    paths={'before':Path('outputs/text_expansion22k_conservative_20261102/training/best_flow.pt'),
           'best2000':RUN/'training/best_flow.pt','latest6000':RUN/'training/latest.pt'}
    reports={};numerics=[]
    for name,path in paths.items():
        state=load_checkpoint(path);model.load_state_dict(state['model']);model.eval()
        torch.manual_seed(123);x=torch.randn(18,4,16,16,device=device);t=torch.linspace(.02,.98,18,device=device)
        with torch.inference_mode():
            context=model.context(text,mask,None,None);torch.backends.mha.set_fastpath_enabled(True)
            fp=model(x,t,context)
            torch.backends.mha.set_fastpath_enabled(False);slow=model(x,t,model.context(text,mask,None,None));torch.backends.mha.set_fastpath_enabled(True)
            model.train();trainmode=model(x,t,model.context(text,mask,None,None));model.eval()
            with torch.autocast('cuda',dtype=torch.bfloat16):bf=model(x,t,model.context(text,mask,None,None)).float()
            numerics.append({'checkpoint':name,'fast_slow_max':(fp-slow).abs().max().item(),'eval_train_max':(fp-trainmode).abs().max().item(),
                             'fp32_bf16_mae':(fp-bf).abs().mean().item(),'fp32_bf16_relative_l2':((fp-bf).norm()/fp.norm()).item()})
        variants=[('fp32_cfg2_32',32,2.,False,True),('fp32_cfg2_128',128,2.,False,True)]
        if name=='best2000':variants += [('fp32_cfg2_20',20,2.,False,True),('fp32_cfg2_64',64,2.,False,True),
            ('fp32_cfg1_32',32,1.,False,True),('fp32_cfg4_32',32,4.,False,True),
            ('bf16_cfg2_32',32,2.,True,True),('fp32_slow_cfg2_32',32,2.,False,False)]
        if name=='latest6000':variants += [('bf16_cfg2_32',32,2.,True,True),('fp32_cfg1_32',32,1.,False,True)]
        for label,steps,scale,bfloat,fast in variants:
            folder=ROOT/name/label;folder.mkdir(parents=True,exist_ok=True);begin=time.time();torch.backends.mha.set_fastpath_enabled(fast)
            with torch.inference_mode(), (torch.autocast('cuda',dtype=torch.bfloat16) if bfloat else nullcontext()):
                raw=sample(model,text,mask,seeds=seeds,steps=steps,text_scale=scale,solver='heun')
            torch.backends.mha.set_fastpath_enabled(True)
            images=[decode(v) for v in raw];rows=[stats(a) for a in images]
            p=((raw.float()+1)/2).clamp(0,1);visible=p[:,3]>.03
            violations=(p[:,:3]>p[:,3:4]+1/255).any(1)
            report={'mean':means(rows),'seconds':time.time()-begin,'premul_violation_visible_fraction':(violations&visible).sum().item()/max(1,visible.sum().item())}
            reports[name+'/'+label]=report
            for i,a in enumerate(images):Image.fromarray(a).save(folder/f'{i:02d}.png')
            torch.save(raw.cpu(),folder/'raw.pt');atomic_json(ROOT/'sampling_report.json',reports)
            print(name,label,report,flush=True)
    atomic_json(ROOT/'forward_numerics.json',numerics)
    atomic_json(ROOT/'cases.json',cases)
    cols=['fp32_cfg2_20','fp32_cfg2_32','fp32_cfg2_64','fp32_cfg2_128','fp32_cfg1_32','fp32_cfg4_32','bf16_cfg2_32']
    sheet=Image.new('RGB',(1320,50+18*110),'white');draw=ImageDraw.Draw(sheet)
    for col,label in enumerate(cols):draw.text((260+col*150,8),label.replace('_','\n'),fill='black')
    for i,c in enumerate(cases):
        y=50+i*110;draw.text((6,y+25),c['evaluation_prompt'],fill='black')
        for col,label in enumerate(cols):
            a=np.array(Image.open(ROOT/'best2000'/label/f'{i:02d}.png'))
            sheet.paste(Image.fromarray(composite(a,128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),(260+col*150,y))
    sheet.save(ROOT/'sampling_comparison.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['data','gpu']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    data() if a.stage=='data' else gpu(a.device)
