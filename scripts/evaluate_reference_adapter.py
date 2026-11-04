"""Auditable fresh train/test qualitative panels for frozen-backbone adaptation."""
import argparse
from collections import Counter,defaultdict
import gc
import html
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import torch.nn.functional as F

from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.images import content_hash,decode,encode
from mcgen.model import PixelFlow
from mcgen.reference_adapter import backbone_digest
from mcgen.resource_text import extract_resource_text
from mcgen.runtime import Runtime,load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from zero_shot_reference import load_jsonl

RUN=Path('outputs/reference_adapter_20261104')
KINDS=['sword','pickaxe','shield','gem','axe','helmet','ingot','bow','potion','apple','boots','shovel']
ALIASES={'crystal':'gem','diamond':'gem','ruby':'gem','sapphire':'gem','emerald':'gem','bottle':'potion',
         'dagger':'sword','katana':'sword','crossbow':'bow','hoe':'shovel','nugget':'ingot'}


def kind(prompt):
    for word in reversed(prompt.split()):
        if word in KINDS:return word
        if word in ALIASES:return ALIASES[word]
    return 'other'


def prepare(out,seed,per_split=12,exclude_manifests=()):
    import pyarrow.parquet as pq
    if per_split<1:raise ValueError('per_split must be positive')
    out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'native').mkdir()
    excluded=set()
    for manifest in exclude_manifests:
        excluded.update(r['target_hash'] for r in json.load(open(manifest)))
    training=json.load(open(RUN/'pairs.json'))
    train_hashes=set();train_groups=set();train_lineages=set();train_prompts=set()
    for r in load_jsonl('outputs/qwen_vl_embedding_large_20261103/catalog/samples.jsonl'):
        if r['split']!='train':continue
        train_hashes.add(r['target_hash']);train_groups.add(r['split_group'])
        train_lineages.update(r.get('source_lineages',[r['lineage']]))
        train_prompts.add(r['prompt'])
    ref_hashes={r['reference_sha256'] for r in training if r['split']=='train'}
    train_pool=defaultdict(list)
    for r in training:
        if r['split']=='train':train_pool[r['target_hash']].append(r)
    meta=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=['file_name','mod_slug']).to_pydict()
    paired=pq.read_table('pairs/stylizer_v4/metadata.parquet').to_pylist();paired={r['index']:r for r in paired}
    manifests={};test_pool=defaultdict(list);rejected=Counter();maps={}
    for r in load_jsonl('data/catalogs/native16_v3/catalog/samples.jsonl'):
        if r['split']!='test' or r['asset_type']!='item' or not r.get('reference'):continue
        if r['target_hash'] in train_hashes or r['split_group'] in train_groups or r['lineage'] in train_lineages:
            rejected['training_overlap']+=1;continue
        name=meta['file_name'][r['source_row']];mod=meta['mod_slug'][r['source_row']]
        policy=extract_resource_text(name,mod)
        if policy['decision']!='keep':rejected['filename_policy_quarantine']+=1;continue
        p=policy['text']
        if not re.fullmatch(r'[a-z]+(?: [a-z]+){0,7}',p):rejected['display_caption_filter']+=1;continue
        origin=paired[r['reference']['index']];src=Path(origin['src'])
        assert int(origin['row'])==r['source_row']
        if str(src) not in manifests:manifests[str(src)]={int(x['row']):x for x in load_jsonl(src/'manifest.jsonl')}
        path=(src/manifests[str(src)][int(origin['row'])]['hd_png']).resolve()
        test_pool[r['target_hash']].append({**r,'prompt':p,'source_filename':name,'source_mod':mod,
                       'reference_file':str(path),'reference_generator':r['generator']})
    rng=random.Random(seed);samples=[];counts={};selection_rejections=Counter()
    for split,pool in [('train',train_pool),('test',test_pool)]:
        buckets=defaultdict(list)
        for h,rows in sorted(pool.items()):buckets[kind(rows[0]['prompt'])].append((h,rows))
        counts[split]={k:len(v) for k,v in buckets.items()}
        for bucket in buckets.values():rng.shuffle(bucket)
        chosen=set(excluded);projects=Counter();selected=[]
        order=KINDS+['other']
        # Preset label-stratified sampling; no generation has occurred yet.
        for round_index in range(per_split+2):
            for category in order:
                if len(selected)==per_split:break
                for h,rows in buckets[category]:
                    if h in chosen:continue
                    if round_index<2 and projects[rows[0]['split_group']]>=3:continue
                    candidates=list(rows);rng.shuffle(candidates)
                    row=None
                    for candidate in candidates:
                        digest=file_hash(candidate['reference_file'])
                        if split=='test' and digest in ref_hashes:
                            selection_rejections['test_reference_hash_in_training']+=1;continue
                        row={**candidate,'reference_sha256':digest};break
                    if row is None:continue
                    spec=row['target'];key=spec['path']
                    if key not in maps:maps[key]=np.memmap(key,mode='r',dtype='uint8',shape=tuple(spec['shape']))
                    pixels=np.array(maps[key][spec['index']]);assert content_hash(pixels)==h
                    if np.count_nonzero(pixels[...,3]>8)<8:
                        chosen.add(h);selection_rejections['near_empty_target']+=1;continue
                    chosen.add(h);projects[row['split_group']]+=1
                    ident=f'{split}_{len(selected):02d}';row.update(case_id=ident,category=category,
                            prompt_seen_in_base_training=row['prompt'] in train_prompts,
                            selection_seed=seed,generation_seed=202611042)
                    alternatives=[]
                    for alternative in candidates:
                        if alternative['reference_file']==row['reference_file']:continue
                        alternate_hash=file_hash(alternative['reference_file'])
                        if split=='test' and alternate_hash in ref_hashes:continue
                        alternatives.append({'reference_file':alternative['reference_file'],
                            'reference_sha256':alternate_hash,'reference_generator':alternative['reference_generator']})
                    row['alternate_references']=alternatives
                    Image.fromarray(pixels).save(out/'inputs'/f'{ident}_target.png')
                    with Image.open(row['reference_file']) as im:im.convert('RGBA').save(out/'inputs'/f'{ident}_reference.png')
                    selected.append(row);break
            if len(selected)==per_split:break
        if len(selected)!=per_split:raise ValueError(f'only {len(selected)} eligible {split} targets')
        samples.extend(selected)
    assert len({r['target_hash'] for r in samples})==2*per_split
    assert not excluded.intersection(r['target_hash'] for r in samples)
    atomic_json(out/'samples.json',samples)
    atomic_json(out/'protocol.json',{'selection_seed':seed,'counts':{'train':per_split,'test':per_split},'candidate_targets_by_kind':counts,
               'excluded_manifests':[str(p) for p in exclude_manifests],'excluded_previous_targets':len(excluded),
               'test_rejections':dict(rejected),'selection_rejections':dict(selection_rejections),
               'test_checks':{'target_absent_from_base_training':True,'project_absent_from_base_training':True,
                              'lineage_absent_from_base_training':True,'reference_file_absent_from_adapter_training':True},
               'caption_policy':'existing filename-only resource_text_v1; no visual labels',
               'sampling':'fixed-seed one target per available declared kind, then fill; one original HD generator variant per target; no output-quality filtering',
               'primary_checkpoint':'best_flow.pt selected by validation before testing',
               'comparison_checkpoints':['best_flow.pt','latest.pt','best_joint_flow.pt'],
               'steps':32,'solver':'heun','guidance':2.,'generation_seeds':[202611042,202611043],
               'test_use':'Post-training qualitative sampling. These viewed samples are no longer an untouched future benchmark. No checkpoint is selected from test outputs.',
               'interpretation':'Illustrative label-stratified samples, not a population-weighted quality score.',
               'training_pairs_sha256':file_hash(RUN/'pairs.json'),'samples_sha256':file_hash(out/'samples.json')})
    for r in samples:print(r['case_id'],r['prompt'],r['reference_generator'],flush=True)


@torch.no_grad()
def diagnostics(model,features,targets,cases,device):
    result={}
    for split in ('train','test'):
        ids=[i for i,r in enumerate(cases) if r['split']==split];x=targets[ids].to(device)
        for mode in ('image','joint'):
            h=features[mode][ids].to(device);wrong=features['wrong_joint'][ids].to(device) if mode=='joint' else h.roll(1,0)
            for j,t_value in enumerate((.1,.5,.9)):
                rng=torch.Generator(device=device).manual_seed(104729+j)
                noise=torch.randn(x.shape,generator=rng,device=device);noisy=(1-t_value)*x+t_value*noise
                t=torch.full((len(x),),t_value,device=device);scores={}
                for name,z,adapt in [('zero_shot',h,False),('matched',h,True),('wrong_reference',wrong,True)]:
                    mask=torch.ones(z.shape[:2],dtype=torch.bool,device=device)
                    present=torch.ones(len(z),dtype=torch.bool,device=device) if adapt else None
                    pred=model(noisy,t,model.context(z,mask,None,present))
                    scores[name]=F.mse_loss(pred.float(),noise-x).item()
                scores['gain_over_wrong_reference']=scores['wrong_reference']-scores['matched']
                result[f'{split}/{mode}/t{t_value}']=scores
    return result


def generate(out,device):
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.65,device)
    started=time.time();cases=json.load(open(out/'samples.json'));checkpoint=RUN/'training/best_flow.pt'
    runtime=Runtime(checkpoint,device=device);images=[Image.open(out/'inputs'/f"{r['case_id']}_reference.png").copy() for r in cases]
    prompts=conditioned_prompts([r['prompt'] for r in cases],runtime.config)
    features={};features['text'],_=runtime.encoder.encode(prompts,'cpu')
    features['image'],_=runtime.encoder.encode_multimodal(['']*len(cases),images,'cpu')
    features['joint'],_=runtime.encoder.encode_multimodal(prompts,images,'cpu')
    wrong_images=[]
    for split in ('train','test'):
        ids=[i for i,r in enumerate(cases) if r['split']==split]
        wrong_images.extend([images[ids[(j+1)%len(ids)]] for j in range(len(ids))])
    features['wrong_joint'],_=runtime.encoder.encode_multimodal(prompts,wrong_images,'cpu')
    torch.save(features,out/'embeddings.pt');identity=runtime.encoder.embedding.multimodal_identity
    del runtime.encoder;gc.collect();torch.cuda.empty_cache()
    source=load_checkpoint('checkpoints/mcgen-step4128.pt');cfg=Experiment.from_dict(source['config'])
    original=PixelFlow(cfg.model).to(device);original.load_state_dict(source['model']);original.eval()
    base_hash=backbone_digest(original);assert backbone_digest(runtime.model)==base_hash
    seeds=[r['generation_seed'] for r in cases];mask=torch.ones((len(cases),1),dtype=torch.bool,device=device)
    def save_generation(key,model,mode,adapter=False,seed_offset=0):
        h=features[mode].to(device)
        values=sample(model,h,mask,seeds=[s+seed_offset for s in seeds],steps=32,text_scale=2.,embedding_adapter=adapter)
        for case,v in zip(cases,values):Image.fromarray(decode(v)).save(out/'native'/f"{case['case_id']}_{key}.png")
        print('GENERATED',key,round(time.time()-started,1),flush=True)
        return values
    text=save_generation('text',original,'text')
    with torch.no_grad():retained=sample(runtime.model,features['text'].to(device),mask,seeds=seeds,steps=32,text_scale=2.)
    assert torch.equal(text,retained),'original text generation changed'
    save_generation('zero_image',original,'image');save_generation('zero_joint',original,'joint')
    targets=torch.stack([encode(np.array(Image.open(out/'inputs'/f"{r['case_id']}_target.png"))) for r in cases])
    reports={};checkpoints={}
    for name in ('best_flow','latest','best_joint_flow'):
        path=RUN/'training'/f'{name}.pt';state=load_checkpoint(path)
        assert Experiment.from_dict(state['config']).model==runtime.config.model
        assert state['multimodal_identity']==identity
        runtime.model.load_state_dict(state['model'],strict=True);runtime.model.eval()
        assert backbone_digest(runtime.model)==base_hash
        checkpoints[name]={'step':state['step'],'sha256':file_hash(path)}
        for mode in ('image','joint'):save_generation(f'{name}_{mode}',runtime.model,mode,True)
        if name=='best_flow':
            for mode in ('image','joint'):save_generation(f'{name}_{mode}_seed2',runtime.model,mode,True,1)
            reports['primary_condition_diagnostics']=diagnostics(runtime.model,features,targets,cases,device)
    atomic_json(out/'results.json',{'checkpoints':checkpoints,'primary':'best_flow','multimodal_identity':identity,
                'base_backbone_sha256':base_hash,'all_checkpoint_backbones_exact':True,'primary_text_generation_exact':True,
                'seconds':time.time()-started,'diagnostics':reports,'warning':'Selected small samples; diagnostic Flow MSE is not a perceptual quality or overall accuracy score.'})
    print('COMPLETE',round(time.time()-started,1),flush=True)


def render(out):
    cases=json.load(open(out/'samples.json'));results=json.load(open(out/'results.json'))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    formats={
       'comparison':(['HD reference','MC target','Text only','Zero-shot image','Trained image','Zero-shot joint','Trained joint'],
                     [None,None,'text','zero_image','best_flow_image','zero_joint','best_flow_joint']),
       'checkpoints':(['HD reference','MC target','Best image','Last image','Best joint','Last joint','Joint-best joint'],
                     [None,None,'best_flow_image','latest_image','best_flow_joint','latest_joint','best_joint_flow_joint']),
       'seeds':(['HD reference','MC target','Image seed 1','Image seed 2','Joint seed 1','Joint seed 2'],
                     [None,None,'best_flow_image','best_flow_image_seed2','best_flow_joint','best_flow_joint_seed2'])}
    panels=[]
    for split in ('train','test'):
        rows=[r for r in cases if r['split']==split]
        overview=Image.new('RGB',(1080,990),'#111827');summary_draw=ImageDraw.Draw(overview)
        summary_draw.text((18,14),f'MCGen | {split.upper()} | 12 fixed samples | adapter step {results["checkpoints"]["best_flow"]["step"]}',font=font,fill='white')
        summary_draw.text((18,37),'Original HD / paired MC target / trained image-only / trained image+text. Raw 16x16 outputs.',font=small,fill='#9ca3af')
        for block in range(2):
            for col,label in enumerate(['HD reference','MC target','Image only','Image + text']):
                summary_draw.text((18+block*536+col*126,64),label,font=font,fill='#e5e7eb')
        for i,case in enumerate(rows):
            block=i//6;row=i%6;ident=case['case_id'];y=88+row*147
            paths=[out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png',
                   out/'native'/f'{ident}_best_flow_image.png',out/'native'/f'{ident}_best_flow_joint.png']
            for col,path in enumerate(paths):
                with Image.open(path) as im:
                    im=im.convert('RGBA');bg=Image.new('RGBA',im.size,(239,241,245,255));bg.alpha_composite(im)
                    tile=bg.convert('RGB').resize((112,112),Image.Resampling.LANCZOS if col==0 else Image.Resampling.NEAREST)
                overview.paste(tile,(18+block*536+col*126,y))
            summary_draw.text((18+block*536,y+117),f'{ident}: {case["prompt"]}',font=small,fill='#e5e7eb')
        overview_name=f'{split}_overview.png';overview.save(out/overview_name);panels.append(overview_name)
        for kind_name,(labels,keys) in formats.items():
            for page in range(2):
                width=22+len(keys)*142;canvas=Image.new('RGB',(width,1080),'#111827');draw=ImageDraw.Draw(canvas)
                draw.text((18,14),f'MCGen reference adapter | {split.upper()} | best={results["checkpoints"]["best_flow"]["step"]} | {kind_name} | {page+1}/2',font=font,fill='white')
                for col,label in enumerate(labels):draw.text((18+col*142,42),label,font=font,fill='#e5e7eb')
                for row,case in enumerate(rows[page*6:page*6+6]):
                    ident=case['case_id'];y=70+row*167
                    for col,key in enumerate(keys):
                        path=(out/'inputs'/f'{ident}_{"reference" if col==0 else "target"}.png') if key is None else out/'native'/f'{ident}_{key}.png'
                        with Image.open(path) as im:
                            im=im.convert('RGBA');bg=Image.new('RGBA',im.size,(239,241,245,255));bg.alpha_composite(im)
                            tile=bg.convert('RGB').resize((128,128),Image.Resampling.LANCZOS if col==0 else Image.Resampling.NEAREST)
                        canvas.paste(tile,(18+col*142,y))
                    caption=f'{ident} | {case["prompt"]} | {case["reference_generator"]}'
                    draw.text((18,y+133),caption,font=small,fill='#e5e7eb')
                name=f'{split}_{kind_name}_{page+1}.png';canvas.save(out/name);panels.append(name)
    body=''.join(f'<figure><img loading="lazy" src="{p}" style="max-width:100%"><figcaption>{html.escape(p)}</figcaption></figure>' for p in panels)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MCGen reference adapter train/test</title><body style="background:#111827;color:#eee;font-family:system-ui"><h1>Train / held-out test samples</h1><p>Raw native 16×16 outputs, enlarged with nearest-neighbor. Main checkpoint selected using validation, before viewing these test generations.</p>'+body+'</body>')
    print('PANELS READY',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    p.add_argument('--phase',choices=['prepare','generate','render','all'],default='all');p.add_argument('--seed',type=int,default=202611041);a=p.parse_args()
    if a.phase in ('prepare','all'):prepare(a.out,a.seed)
    if a.phase in ('generate','all'):generate(a.out,a.device)
    if a.phase in ('render','all'):render(a.out)
