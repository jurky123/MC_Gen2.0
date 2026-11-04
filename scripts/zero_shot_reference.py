"""Frozen step4128 generator: compare text / HD image / image+text embeddings.

No optimizer, parameter update, captioning, reference CNN or target-image encoding.
Uses the original HD PNGs, not the 128px spatial-reference cache. Validation
targets stay in the existing split; test data are never selected.
"""
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.images import content_hash,decode
from mcgen.runtime import Runtime
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts


def load_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def collect_pairs():
    import pyarrow.parquet as pq
    clean={}
    for r in load_jsonl('outputs/resource_text_training_20261103/catalog/samples.jsonl'):
        if r['split'] in ('train','val'):clean.setdefault(r['target_hash'],r)
    metadata={r['index']:r for r in pq.read_table('pairs/stylizer_v4/metadata.parquet').to_pylist()}
    manifests={}
    groups=defaultdict(dict)
    counts=Counter()
    for r in load_jsonl('data/catalogs/native16_v3/catalog/samples.jsonl'):
        c=clean.get(r['target_hash'])
        if not c or r['split']!=c['split'] or r['asset_type']!='item' or not r.get('reference'):continue
        m=metadata[r['reference']['index']]
        src=Path(m['src'])
        if str(src) not in manifests:manifests[str(src)]={int(x['row']):x for x in load_jsonl(src/'manifest.jsonl')}
        original=manifests[str(src)][int(m['row'])]
        path=(src/original['hd_png']).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        counts[c['split']+'_pairs']+=1
        groups[r['target_hash']][r['generator']]={**r,'prompt':c['prompt'],'category':c.get('category_15','other'),
                    'reference_file':str(path),'original_manifest':str(src/'manifest.jsonl'),
                    'source_filename':c.get('source_filename',''),'reference_generator':r['generator']}
    return groups,counts


def prepare(out):
    groups,counts=collect_pairs()
    cases=[]
    # Freeze deterministic, varied train/validation examples before seeing outputs.
    preferred=['sword','pickaxe','shield','gem','axe','food','helmet','ingot','bottle','chestplate','other']
    for split in ('train','val'):
        pool=[g for h,g in sorted(groups.items()) if len(g)==2 and next(iter(g.values()))['split']==split]
        counts[split+'_targets']=sum(next(iter(g.values()))['split']==split for g in groups.values())
        selected=[]
        for category in preferred:
            candidates=[g for g in pool if next(iter(g.values()))['category']==category and g not in selected]
            if candidates:selected.append(min(candidates,key=lambda g:(len(next(iter(g.values()))['prompt']),next(iter(g.values()))['target_hash'])))
            if len(selected)==6:break
        for g in pool:
            if len(selected)==6:break
            if g not in selected:selected.append(g)
        for g in selected:
            for generator,r in sorted(g.items()):
                number=len(cases);case_id=f'{number:02d}_{split}_{generator}'
                spec=r['target'];target=np.array(np.memmap(spec['path'],mode='r',dtype='uint8',shape=tuple(spec['shape']))[spec['index']])
                assert content_hash(target)==r['target_hash']
                Image.fromarray(target).save(out/'inputs'/f'{case_id}_target.png')
                with Image.open(r['reference_file']) as im:im.convert('RGBA').save(out/'inputs'/f'{case_id}_reference.png')
                cases.append({**r,'case_id':case_id,'reference_sha256':file_hash(r['reference_file']),
                              'seed':20261104,'selected_before_generation':True})
    assert len(cases)==24, len(cases)
    atomic_json(out/'cases.json',cases)
    atomic_json(out/'data_audit.json',{'available_clean_pairs':dict(counts),'cases':len(cases),'selected_targets':12,
                'splits':['train','val'],'test_used':False,'reference_input':'original HD PNG; alpha composited white',
                'selection':'six categories per split, shortest cleaned filename prompt then hash; two generators per target'})
    print('PREPARED',dict(counts),[(r['split'],r['prompt'],r['reference_generator']) for r in cases],flush=True)


def official_check(runtime,out,source,images,prompts):
    spec=importlib.util.spec_from_file_location('official_reference',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    impl=runtime.encoder.embedding
    class Adapter:
        device=impl.device
        def __call__(self,**inputs):
            return SimpleNamespace(last_hidden_state=impl.model(**inputs,use_cache=False).last_hidden_state,attention_mask=inputs['attention_mask'])
    ref=object.__new__(module.Qwen3VLEmbedder)
    ref.model=Adapter();ref.processor=impl.processor;ref.default_instruction="Represent the user's input."
    ref.max_length=512;ref.min_pixels=4096;ref.max_pixels=512*512;ref.total_pixels=7864320;ref.fps=1;ref.max_frames=64
    from mcgen.qwen_embedding import embedding_image
    checks={}
    for mode,texts in [('image_only',['','']),('image_text',prompts[:2])]:
        canonical=[embedding_image(x) for x in images[:2]]
        got,_=runtime.encoder.encode_multimodal(texts,canonical,runtime.device)
        with torch.inference_mode():expected=torch.cat([ref.process([{'image':im,**({'text':p} if p else {})}]) for im,p in zip(canonical,texts)])
        cosine=torch.nn.functional.cosine_similarity(got[:,0].float(),expected.float())
        checks[mode]={'minimum_cosine':cosine.min().item(),'max_absolute_difference':(got[:,0].float()-expected.float()).abs().max().item()}
        assert cosine.min().item()>.99999,checks
    # Prove that extending image support retained the accepted text protocol.
    got,_=runtime.encoder.encode(prompts[:2],runtime.device)
    with torch.inference_mode():expected=ref.process([{'text':p} for p in prompts[:2]])
    checks['text']={'minimum_cosine':torch.nn.functional.cosine_similarity(got[:,0].float(),expected.float()).min().item(),
                    'max_absolute_difference':(got[:,0].float()-expected.float()).abs().max().item()}
    assert checks['text']['minimum_cosine']>.99999
    atomic_json(out/'protocol_check.json',{'official_source_sha256':file_hash(source),'checks':checks,
                 'text_identity':runtime.encoder.identity,'multimodal_identity':impl.multimodal_identity})
    print('OFFICIAL PROTOCOL CHECK',checks,flush=True)


def run(args,out):
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.65,args.device)
    started=time.time()
    runtime=Runtime(args.checkpoint,device=args.device)
    cases=json.load(open(out/'cases.json'))
    images=[Image.open(out/'inputs'/f"{r['case_id']}_reference.png").copy() for r in cases]
    prompts=conditioned_prompts([r['prompt'] for r in cases],runtime.config)
    official_check(runtime,out,args.official_source,images,prompts)
    features={}
    for mode in ('text','image','joint'):
        chunks=[]
        for start in range(0,len(cases),4):
            texts=prompts[start:start+4]
            if mode=='text':h,_=runtime.encoder.encode(texts,'cpu')
            else:h,_=runtime.encoder.encode_multimodal(['']*len(texts) if mode=='image' else texts,images[start:start+4],'cpu')
            chunks.append(h)
        features[mode]=torch.cat(chunks)
        print('ENCODED',mode,round(time.time()-started,1),flush=True)
    torch.save(features,out/'embeddings.pt')
    # Unload the large frozen encoder once the audited embeddings are cached.
    del runtime.encoder
    import gc
    gc.collect();torch.cuda.empty_cache()
    n=len(cases);base_seed=cases[0]['seed']
    variants=[('text',2.,base_seed),('image',1.,base_seed),('image',2.,base_seed),('image',3.,base_seed),
              ('image',2.,base_seed+1),('joint',2.,base_seed),('null',0.,base_seed)]
    keys=[]
    for mode,scale,seed in variants:
        key=f'{mode}_cfg{scale:g}_seed{seed}';keys.append(key)
        h=features['image' if mode=='null' else mode].to(args.device)
        mask=torch.full(h.shape[:2],mode!='null',dtype=torch.bool,device=args.device)
        with torch.inference_mode():
            result=sample(runtime.model,h,mask,seeds=[seed]*n,steps=args.steps,text_scale=scale)
        for case,x in zip(cases,result):Image.fromarray(decode(x)).save(out/'native'/f"{case['case_id']}_{key}.png")
        print('GENERATED',key,round(time.time()-started,1),flush=True)
    img=features['image'][:,0].float();txt=features['text'][:,0].float()
    similarity=img@txt.T
    # Texts repeat for FLUX/Qwen. Compare target identity rather than row index.
    top=similarity.argmax(1).tolist()
    diagnostic={'paired_text_image_cosine_mean':similarity.diag().mean().item(),
                'other_target_text_cosine_mean':float(np.mean([similarity[i,j].item() for i in range(n) for j in range(n) if cases[i]['target_hash']!=cases[j]['target_hash']])),
                'within_panel_retrieval_top1_target_accuracy':sum(cases[i]['target_hash']==cases[j]['target_hash'] for i,j in enumerate(top))/n,
                'note':'Small selected panel diagnostics, not a population quality or generalization score.'}
    atomic_json(out/'results.json',{'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':file_hash(args.checkpoint),
                 'step':runtime.metadata['step'],'training_updates':0,'parameter_updates':False,'steps':args.steps,
                 'variants':keys,'seconds':time.time()-started,'diagnostics':diagnostic,
                 'guidance':'ref-scale guides the complete image/text embedding through the existing text projection',
                 'limitations':'Untrained modality transfer; pooled retrieval embeddings need not match the text condition distribution.'})
    print('COMPLETE',diagnostic,flush=True)


def render(out):
    cases=json.load(open(out/'cases.json'));result=json.load(open(out/'results.json'))
    variants=[x for x in result['variants'] if not x.startswith('null')]
    labels=['Original HD','MC target','Text only','Image CFG 1','Image CFG 2','Image CFG 3','Image seed 2','Image + text']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    def tile(image):
        image=image.convert('RGBA')
        bg=Image.new('RGBA',image.size,(239,241,245,255));bg.alpha_composite(image)
        return bg.convert('RGB')
    for page in range(4):
        canvas=Image.new('RGB',(1168,1140),'#111827');draw=ImageDraw.Draw(canvas)
        split=cases[page*6]['split']
        draw.text((20,15),f'MCGen step 4128 | zero additional training | {split.upper()} | panel {page+1}/4',font=font,fill='white')
        draw.text((20,40),'Same seed across columns; original 16x16 RGBA output enlarged with nearest-neighbor.',font=small,fill='#9ca3af')
        for c,label in enumerate(labels):draw.text((20+c*142,70),label,font=font,fill='#e5e7eb')
        for row,case in enumerate(cases[page*6:page*6+6]):
            y=100+row*170;ident=case['case_id']
            paths=[out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png']+[out/'native'/f'{ident}_{key}.png' for key in variants]
            for c,path in enumerate(paths):
                with Image.open(path) as im:
                    picture=tile(im)
                    if c==0:picture.thumbnail((128,128),Image.Resampling.LANCZOS)
                    else:picture=picture.resize((128,128),Image.Resampling.NEAREST)
                    canvas.paste(picture,(20+c*142+(128-picture.width)//2,y+(128-picture.height)//2))
            draw.text((20,y+134),f"{ident} | {case['prompt']} | {case['category']}",font=small,fill='#e5e7eb')
        canvas.save(out/f'comparison_{page+1}.png')
    print('PANELS READY',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--checkpoint',default='checkpoints/mcgen-step4128.pt')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--official-source',type=Path,default=Path('outputs/qwen_vl_embedding_large_20261103/official_embedding_reference.py'))
    p.add_argument('--phase',choices=('prepare','run','render','all'),default='all')
    args=p.parse_args();out=args.out
    if args.phase in ('prepare','all'):
        out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'native').mkdir();prepare(out)
    if args.phase in ('run','all'):run(args,out)
    if args.phase in ('render','all'):render(out)


if __name__=='__main__':main()
