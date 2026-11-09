"""Prepare full native16 supervision and an immutable caption-token cache plan."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import yaml

from mcgen.catalog import atomic_json,file_hash,read_catalog,write_catalog
from mcgen.config import Experiment
from mcgen.dataset import CatalogDataset
from mcgen.reference_recovery import coarse_caption
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts
from mcgen.text_replay import select_replay_records,TextReplayData
from mcgen.qwen_text_tokens import token_identity,tokenize_captions
from prepare_reference_recovery import COMMON

DEFAULT=Path('outputs/token_full_hq_20261109')
SOURCE=Path('outputs/reference_text_replay2000_20261108_balanced')
CATALOG=Path('outputs/qwen_vl_embedding_large_20261103/catalog')


def prepare(out):
    from transformers import AutoProcessor
    out=out.resolve()
    if out.exists():raise ValueError('Use a fresh token experiment directory')
    out.mkdir(parents=True)
    for name in ('pairs.json','review_snapshot.json','data_audit.json','excluded_pairs.json'):
        shutil.copy2(SOURCE/name,out/name)
    (out/'visual_features').symlink_to((SOURCE/'visual_features').resolve(),target_is_directory=True)
    pairs=json.loads((out/'pairs.json').read_text())
    cfg=Experiment.load(SOURCE/'config.yaml')
    cfg.text.backend='qwen3_vl_tokens';cfg.model.text_length=512
    cfg.train.steps=12000;cfg.train.batch_size=640;cfg.train.accumulation=1
    cfg.train.lr=1e-6;cfg.train.warmup=500
    cfg.train.task_mix={'joint':124/640,'text':412/640,'reference':62/640,'unconditional':42/640}
    cfg=Experiment.from_dict(cfg.to_dict())
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
    checkpoint=SOURCE/'training/best_text.pt';state=load_checkpoint(checkpoint)
    if state.get('diagnostic_only') or state.get('target_leakage'):raise ValueError('Invalid source checkpoint')
    source_identity=state['text_identity'];visual_identity=state['visual_identity'];source_step=state['step']
    del state;gc.collect()
    records,parent=read_catalog(CATALOG)
    selected,audit=select_replay_records(records,pairs)
    del records
    meta=write_catalog(out/'full_text_catalog',selected,{'native_size':16,
        'parent_fingerprint':parent['fingerprint'],'supervision':'unchanged original filename text',
        'selection':'all eligible original train/val; exclude paired targets and reference holdout projects'})
    train=CatalogDataset(selected,16,16,'train',task='text')
    val=CatalogDataset(selected,16,16,'val',task='text')
    chooser=TextReplayData.__new__(TextReplayData);chooser.datasets={'val':val}
    val_ids=chooser.validation_indices(1024)
    rng=random.Random(2026110931)
    train_ids=rng.sample(range(len(train)),512)
    probe_records=[train.records[i] for i in train_ids]+[val.records[i] for i in val_ids]
    write_catalog(out/'probe_catalog',probe_records,{'native_size':16,'purpose':'disposable real-data checks; no training subset restriction'})
    required=[p for row in pairs for p in row['captions']]
    required += [coarse_caption(row['captions'][0]) for row in pairs]+COMMON+['']
    required += [r['prompt'] for r in probe_records]
    priority=sorted(set(conditioned_prompts(required,cfg)))
    full=sorted(set(conditioned_prompts([r['prompt'] for r in selected],cfg)) - set(priority))
    prompts=priority+full
    cache=out/'text_features';cache.mkdir()
    atomic_json(cache/'prompts.json',prompts)
    model_root=Path(os.environ.get('MCGEN_EMBEDDING_MODEL') or cfg.text.model)
    manifest=json.loads((model_root/'verified_download.json').read_text())
    if not manifest['complete'] or manifest['revision']!=cfg.text.revision:raise ValueError('Wrong Qwen artifacts')
    identity=token_identity(cfg.text,cfg.model,{r['name']:r['verified_sha256'] for r in manifest['files']})
    processor=AutoProcessor.from_pretrained(model_root,local_files_only=True,padding_side='right')
    input_offsets=[0];offsets=[0];lengths=[]
    with (cache/'input_ids.int32.mmap').open('wb') as f,(cache/'positions.uint16.mmap').open('wb') as g:
        for start in range(0,len(prompts),256):
            rows=tokenize_captions(processor,prompts[start:start+256],cfg.text.instruction,cfg.model.text_length)
            for row in rows:
                ids,positions=row['input_ids'],row['positions']
                np.asarray(ids,dtype='int32').tofile(f);np.asarray(positions,dtype='uint16').tofile(g)
                input_offsets.append(input_offsets[-1]+len(ids));offsets.append(offsets[-1]+len(positions));lengths.append(len(positions))
            if start%8192==0:print('TOKENIZED',start+len(rows),len(prompts),flush=True)
    np.asarray(offsets,dtype='uint64').tofile(cache/'offsets.uint64.mmap')
    np.asarray(input_offsets,dtype='uint64').tofile(cache/'input_offsets.uint64.mmap')
    shape=[offsets[-1],4096]
    with (cache/'features.uint16.mmap').open('wb') as f:f.truncate(offsets[-1]*4096*2)
    plan={'format':'ragged_bfloat16_text_v1','complete':False,'shape':shape,
        'count':len(prompts),'max_tokens':512,'actual_max_tokens':max(lengths),
        'text_identity':identity,'prompts_sha256':file_hash(cache/'prompts.json'),
        'offsets_sha256':file_hash(cache/'offsets.uint64.mmap'),
        'input_offsets_sha256':file_hash(cache/'input_offsets.uint64.mmap'),
        'input_ids_sha256':file_hash(cache/'input_ids.int32.mmap'),
        'positions_sha256':file_hash(cache/'positions.uint16.mmap'),
        'chunk_prompts':512,'probe_prompt_count':len(priority),
        'dtype':'bfloat16','max_input_tokens':512,'canonical_forward':'singleton',
        'token_length_quantiles':{str(q):float(np.quantile(lengths,q)) for q in (0,.5,.9,.99,1)},
        'features_bytes':offsets[-1]*4096*2}
    atomic_json(cache/'plan.json',plan);atomic_json(cache/'metadata.json',plan)
    (cache/'chunks').mkdir();(cache/'locks').mkdir()
    recipe={'version':'token_full_hq_v1','steps':12000,'batch_size':640,'pairs_per_update':62,
        'unconditional_per_update':10,'replay_text_per_update':288,'replay_null_per_update':32,
        'source_hq_fraction':.8,'full_text_data':True,'full_text_train_targets':len(train.groups[False]),
        'base_lr':1e-6,'visual_lr':1e-5,'text_lr':1e-5,'warmup_steps':500,
        'decay_start':8000,'lr_floor':.2,'gradient_clip':1.,'freeze_time_mlp':True,
        'length_sorted_microbatches':True,'gpu_memory_fraction':.85,
        'evaluate_updates':[0,250,500,1000,2000,4000,8000,12000],
        'broad_text_generation_samples':48,'broad_text_flow_samples':1024,
        'gallery_name':'token-full','evaluation_gallery_name':'token-full-evaluation',
        'joint_text_control_limitation':'No edit/counterfactual targets added',
        'stability_guard':{'warmup':500,'calibration':400,'window':100,'gradient_clip':1.,
            'gradient_ratio':5.,'loss_ratio':1.15,'clipped_fraction':.8,'patience':3}}
    atomic_json(out/'recipe.pending.json',recipe)
    atomic_json(out/'initialization.json',{'checkpoint':str(checkpoint.resolve()),'sha256':file_hash(checkpoint),
        'source_step':source_step,'source_run':str(SOURCE.resolve()),'visual_identity':visual_identity,
        'transfer':'strict learned weights; new token condition identity; fresh optimizer',
        'text_protocol_transition':{'source_identity':source_identity,'target_identity':identity},
        'freeze_time_mlp':True,'diagnostic_weights_used':False})
    info={'preparation_complete':True,'train_text_targets':len(train.groups[False]),'val_text_targets':len(val),
        'reference_train_pairs':sum(r['split']=='train' for r in pairs),
        'reference_train_unique':len({r['target_hash'] for r in pairs if r['split']=='train'}),
        'hq_target_sampling_fraction':.8,'hq_train_unique':len({r['target_hash'] for r in pairs if r['split']=='train' and r['quality_tier']=='highest_quality'}),
        'source_catalog_fingerprint':parent['fingerprint'],'catalog_fingerprint':meta['fingerprint'],
        'audit':audit,'token_cache':plan,'steps':12000,'logical_batch':640,
        'broad_conditional_exposures_per_target':12000*288/len(train.groups[False]),
        'text_only_fraction':412/640,'cache_complete':False,'long_training_started':False}
    atomic_json(out/'preparation.json',info)
    print('PREPARED',json.dumps({k:v for k,v in info.items() if k not in ('audit','token_cache')},ensure_ascii=False),flush=True)
    print('TOKEN_CACHE',json.dumps(plan,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    prepare(p.parse_args().out)
