"""Prepare original-HD pairs and canonical singleton frozen embeddings."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts
from zero_shot_reference import collect_pairs


def prepare(out,checkpoint):
    out.mkdir(parents=True,exist_ok=False);(out/'features').mkdir()
    groups,counts=collect_pairs()
    records=[];seen={}
    for target,variants in sorted(groups.items()):
        for gen,r in sorted(variants.items()):
            digest=file_hash(r['reference_file'])
            owner=seen.setdefault(digest,r['split'])
            # Drop cross-split reference collisions from validation. Targets
            # already obey the canonical project/hash split.
            records.append({**r,'reference_sha256':digest,'reference_collision':owner!=r['split']})
    leaked={r['reference_sha256'] for r in records if r['reference_collision']}
    records=[r for r in records if not (r['split']=='val' and r['reference_sha256'] in leaked)]
    for i,r in enumerate(records):r['feature_index']=i
    atomic_json(out/'pairs.json',records)
    state=load_checkpoint(checkpoint);data=state['config']
    data['model']['multimodal_adapter']=True
    data['conditioning_input']='multimodal_embedding';data['task']='joint'
    data['train'].update(trainable='embedding_adapter',steps=2000,batch_size=256,accumulation=1,warmup=100,lr=.0002,
                         eval_every=200,val_samples=256,task_mix={'text':0.,'reference':.5,'joint':.5,'unconditional':0.})
    cfg=Experiment.from_dict(data)
    import yaml
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
    atomic_json(out/'data_audit.json',{'source_counts':dict(counts),'canonical_pairs':dict(Counter(r['split'] for r in records)),
                'unique_targets':{s:len({r['target_hash'] for r in records if r['split']==s}) for s in ('train','val')},
                'selection':'one original HD variant per target/generator; target-uniform training',
                'reference_hashes_excluded_from_val':len(leaked),'test_used':False,
                'pairs_sha256':file_hash(out/'pairs.json'),'base_checkpoint':str(Path(checkpoint).resolve()),
                'base_checkpoint_sha256':file_hash(checkpoint),'base_step':state['step'],'base_text_identity':state['text_identity']})
    print('PREPARED',len(records),flush=True)


def cache(out,device):
    atomic_json(out/'cache_worker.json',{'pid':os.getpid()})
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.65,device)
    cfg=Experiment.load(out/'config.yaml');encoder=TextEncoder(cfg.text,cfg.model,device)
    records=json.load(open(out/'pairs.json'));start=time.time();features=out/'features';processed_this_run=0
    for chunk_start in range(0,len(records),64):
        path=features/f'{chunk_start:06d}.pt'
        if path.exists():
            old=torch.load(path,weights_only=True)
            if old['multimodal_identity']!=encoder.embedding.multimodal_identity or old['pairs_sha256']!=file_hash(out/'pairs.json'):raise ValueError('stale image cache')
            continue
        rows=records[chunk_start:chunk_start+64];images=[]
        for r in rows:
            if file_hash(r['reference_file'])!=r['reference_sha256']:raise ValueError('changed reference image')
            with Image.open(r['reference_file']) as im:images.append(im.convert('RGBA'))
        image,_=encoder.encode_multimodal(['']*len(rows),images,'cpu')
        joint,_=encoder.encode_multimodal(conditioned_prompts([r['prompt'] for r in rows],cfg),images,'cpu')
        tmp=path.with_suffix('.tmp');torch.save({'start':chunk_start,'image':image,'joint':joint,'multimodal_identity':encoder.embedding.multimodal_identity,
                                               'pairs_sha256':file_hash(out/'pairs.json')},tmp);tmp.replace(path)
        processed_this_run+=len(rows)
        elapsed=time.time()-start
        atomic_json(out/'cache_progress.json',{'done':chunk_start+len(rows),'total':len(records),'seconds':elapsed,
                    'processed_this_run':processed_this_run,
                    'estimated_remaining_seconds':elapsed/processed_this_run*max(0,len(records)-chunk_start-len(rows))})
        if chunk_start%512==0:print('CACHED',chunk_start+len(rows),len(records),round(elapsed,1),flush=True)
    atomic_json(features/'metadata.json',{'complete':True,'rows':len(records),'pairs_sha256':file_hash(out/'pairs.json'),
                'text_identity':encoder.identity,'multimodal_identity':encoder.embedding.multimodal_identity,
                'batch_policy':'singleton, identical to runtime','chunks':{p.name:file_hash(p) for p in sorted(features.glob('*.pt'))}})
    print('CACHE COMPLETE',round(time.time()-start,1),flush=True)


def controls(out,device):
    """Hold text fixed while swapping the reference, to measure image benefit."""
    from collections import defaultdict
    from train_reference_adapter import draw
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.65,device)
    cfg=Experiment.load(out/'config.yaml');rows=json.load(open(out/'pairs.json'));groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='val':groups[r['target_hash']].append(i)
    ids=draw(list(groups.values()),cfg.train.val_samples,torch.Generator().manual_seed(7919))
    wrong=[]
    for k,i in enumerate(ids):
        offset=1
        while rows[ids[(k+offset)%len(ids)]]['target_hash']==rows[i]['target_hash']:offset+=1
        wrong.append(ids[(k+offset)%len(ids)])
    encoder=TextEncoder(cfg.text,cfg.model,device);chunks=[]
    for start in range(0,len(ids),16):
        images=[]
        for i in wrong[start:start+16]:
            with Image.open(rows[i]['reference_file']) as im:images.append(im.convert('RGBA'))
        prompts=conditioned_prompts([rows[i]['prompt'] for i in ids[start:start+16]],cfg)
        h,_=encoder.encode_multimodal(prompts,images,'cpu');chunks.append(h)
    torch.save({'ids':ids,'wrong_ids':wrong,'wrong_image_same_text':torch.cat(chunks),
                'multimodal_identity':encoder.embedding.multimodal_identity},out/'validation_controls.pt')
    atomic_json(out/'validation_controls.json',{'sha256':file_hash(out/'validation_controls.pt'),'cases':len(ids),
                 'control':'same text, different target reference; no training on these mismatches'})
    print('VALIDATION CONTROLS READY',len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','controls']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',default='checkpoints/mcgen-step4128.pt');p.add_argument('--device',default='cuda:1');a=p.parse_args()
    if a.stage=='prepare':prepare(a.out,a.checkpoint)
    elif a.stage=='controls':controls(a.out,a.device)
    else:
        try:cache(a.out,a.device)
        except Exception as exc:
            atomic_json(a.out/'cache_failed.json',{'error':str(exc)});raise
