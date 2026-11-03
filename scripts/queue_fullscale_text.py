"""Resumable serial queue: 10k eligible train captions -> exact features -> full training -> paired eval."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from mcgen.catalog import atomic_json, file_hash, read_catalog, write_catalog
from mcgen.caption_supervision import caption_supervision
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts

ROOT = Path('outputs/fullscale_text10k_20261103')
SOURCE = Path('outputs/fullscale_clean_v2_20261102/catalog')
PARENT = Path('outputs/text_expansion22k_conservative_20261102/training/best_flow.pt')
DEVICE = 'cuda:0'


def status(stage, **kwargs):
    row = {'stage': stage, 'updated_unix': time.time(), **kwargs}
    atomic_json(ROOT/'queue_status.json', row)
    print(json.dumps(row), flush=True)


def prepare():
    ROOT.mkdir(exist_ok=False)
    records, meta = read_catalog(SOURCE)
    short = {r['target_hash']: r for r in records if r['caption_variant']=='short' and r['split']=='train'}
    existing = {r['target_hash'] for r in records if r['caption_variant']=='long' and r['split']=='train'}
    assert len(short)==324921 and len(existing)==2037
    candidates = [r for h,r in sorted(short.items()) if h not in existing]
    random.Random(20261103).shuffle(candidates)
    candidates = candidates[:24000]  # Fixed train-only reserve; annotate only as many as needed.
    with (ROOT/'annotation_cases.jsonl').open('w') as f:
        for r in candidates: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    cfg = yaml.safe_load(Path('outputs/text_expansion22k_conservative_20261102/config.yaml').read_text())
    cfg['train'].update(steps=200000, batch_size=32, accumulation=2, warmup=1000,
                        lr=1e-5, eval_every=2000, val_samples=1024, seed=20261103, schedule_start=0)
    cfg['text']['cache_mb']=0
    (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    Experiment.load(ROOT/'config.yaml')
    parent = load_checkpoint(PARENT); assert parent['step']==8000
    atomic_json(ROOT/'protocol.json', {'initialization':str(PARENT),'initialization_sha256':file_hash(PARENT),
        'parent_step':8000,'endpoint':200000,'source_catalog':str(SOURCE),'source_fingerprint':meta['fingerprint'],
        'config_sha256':file_hash(ROOT/'config.yaml'),'annotation_cases_sha256':file_hash(ROOT/'annotation_cases.jsonl'),
        'target_eligible_train_descriptions':10000,'existing_eligible_train_descriptions':len(existing),
        'annotation_reserve':len(candidates),'selection':'Uniform distinct train targets, fixed seed; existing eligible captions retained. No test/val annotation.',
        'optimizer':'Fresh Adam for new data/schedule; initialize model from accepted best step 8000.',
        'text_feature_policy':'Exact Qwen3-8B BF16 layers 9/18/27 at full 128 positions, no padding trimming.',
        'expected_text_draws_per_train_target':200000*64*.9/len(short),
        'estimated_wall_time':'Planning range 1-3 days on shared A100, not a measured guarantee; annotation roughly 1.5-3h; feature/training ETA recorded after progress.',
        'production_promoted':False})
    # Original development probes plus fixed broad cleaned-validation filename probes.
    evaluation=json.load(open('outputs/text_expansion2k_20261102/eval_cases.json'))
    used={r.get('target_hash') for r in evaluation['cases']}
    val=[r for r in records if r['split']=='val' and r['caption_variant']=='short' and r['target_hash'] not in used]
    random.Random(20261103).shuffle(val)
    evaluation['cases'].extend({**r,'evaluation_prompt':r['prompt'],'group':'fullscale_val_filename','seed':20261103} for r in val[:128])
    atomic_json(ROOT/'eval_cases.json',evaluation)
    status('queued', stages=['annotate_to_10000','build_catalog','precompute_text','train_200000','evaluate_final','evaluate_best'])


def eligible_annotations():
    result = {}; path = ROOT/'annotations.jsonl'
    if not path.exists(): return result
    for line in path.open():
        r=json.loads(line)
        policy=caption_supervision(r['source_filename'],r.get('combined_text'),r.get('quality_flags',[]))
        if policy['caption_variants']==['short','long']:result[r['target_hash']]=(r,policy)
    return result


def command(stage, args):
    status(stage)
    with (ROOT/(stage+'.log')).open('a') as f:
        subprocess.run([sys.executable,*args],check=True,stdout=f,stderr=subprocess.STDOUT,
                       env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})


def annotate():
    required=10000-2037
    while len(eligible_annotations())<required:
        accepted=len(eligible_annotations())
        n=sum(1 for _ in (ROOT/'annotations.jsonl').open()) if (ROOT/'annotations.jsonl').exists() else 0
        if n>=24000:raise RuntimeError('Annotation reserve exhausted before 10000 eligible descriptions; training blocked.')
        count=min(24000-n,math.ceil((required-accepted)*1.08)+32)
        command('annotation',['scripts/annotate_text_expansion.py','--root',str(ROOT),'--device',DEVICE,'--limit',str(count)])
    status('annotation_target_reached',new_eligible=len(eligible_annotations()),total_required=10000)


def build():
    if (ROOT/'catalog/catalog.json').exists():
        _,meta=read_catalog(ROOT/'catalog');return
    records,parent=read_catalog(SOURCE)
    assert parent['fingerprint']==json.load(open(ROOT/'protocol.json'))['source_fingerprint']
    accepted=eligible_annotations(); selected=dict(list(accepted.items())[:7963]); assert len(selected)==7963
    added=[]
    for r in records:
        if r['split']=='train' and r['caption_variant']=='short' and r['target_hash'] in selected:
            annotation,policy=selected[r['target_hash']]
            added.append({**r,'id':r['id'].removesuffix(':short')+':long','prompt':policy['descriptive_text'],
                          'caption_variant':'long','caption_supervision':'automatic_weak_label_10k',
                          'annotation_quality_flags':annotation['quality_flags']})
    records.extend(added)
    assert sum(r['split']=='train' and r['caption_variant']=='long' for r in records)==10000
    meta=write_catalog(ROOT/'catalog',records,{'native_size':16,'parent_fingerprint':parent['fingerprint'],
        'annotations_sha256':file_hash(ROOT/'annotations.jsonl'),'protocol_sha256':file_hash(ROOT/'protocol.json'),
        'policy':'Uniform target; filename and eligible description each 50%; bad descriptions retain filename-only images.',
        'automatic_labels_not_human_gold':True,'train_descriptions':10000,'test_used':False})
    atomic_json(ROOT/'caption_build_report.json',{'catalog_fingerprint':meta['fingerprint'],'train_targets':324921,
        'dual_caption_train_targets':10000,'filename_only_train_targets':314921,
        'descriptive_fraction_given_text':10000/324921/2,'new_eligible_generated':len(accepted),
        'new_eligible_used':len(added),'audit':meta['audit']})


def wait_gpu(minimum_gib):
    while torch.cuda.mem_get_info(DEVICE)[0]<minimum_gib*1024**3:
        if (ROOT/'STOP_REQUESTED.json').exists():
            status('stopped_by_request', checkpoint_saved=False);raise SystemExit(0)
        status('waiting_for_gpu',device=DEVICE,required_free_gib=minimum_gib);time.sleep(30)
    free,total=torch.cuda.mem_get_info(DEVICE)
    torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),DEVICE)


def features():
    cfg=Experiment.load(ROOT/'config.yaml'); records,meta=read_catalog(ROOT/'catalog')
    prompts=sorted(set(conditioned_prompts([r['prompt'] for r in records]+[''],cfg)))
    out=ROOT/'text_features';out.mkdir(exist_ok=True)
    shape=(len(prompts),cfg.model.text_length,cfg.model.text_dim)
    if (out/'metadata.json').exists():
        progress=json.load(open(out/'metadata.json'))
        assert progress['catalog_fingerprint']==meta['fingerprint'] and progress['shape']==list(shape)
        assert json.load(open(out/'prompts.json'))==prompts
        if progress['complete']:
            DiskTextEncoder(out,load_checkpoint(PARENT)['text_identity']);return
    else:
        required=int(np.prod(shape))*2+shape[0]*shape[1]
        if shutil.disk_usage(out).free < required+100*1024**3:raise RuntimeError('Insufficient disk space for exact features plus 100GiB reserve')
        atomic_json(out/'prompts.json',prompts)
        progress={'shape':list(shape),'catalog_fingerprint':meta['fingerprint'],'prompts_sha256':file_hash(out/'prompts.json'),
                  'complete':False,'completed_rows':0,'bytes':required}
        np.memmap(out/'features.uint16.mmap',mode='w+',dtype='uint16',shape=shape).flush()
        np.memmap(out/'masks.uint8.mmap',mode='w+',dtype='uint8',shape=shape[:2]).flush()
        atomic_json(out/'metadata.json',progress)
    wait_gpu(23);encoder=TextEncoder(cfg.text,cfg.model,DEVICE)
    assert encoder.identity==load_checkpoint(PARENT)['text_identity']
    progress['text_identity']=encoder.identity
    data=np.memmap(out/'features.uint16.mmap',mode='r+',dtype='uint16',shape=shape)
    mask=np.memmap(out/'masks.uint8.mmap',mode='r+',dtype='uint8',shape=shape[:2])
    start=progress['completed_rows'];begin=time.time()
    for offset in range(start,len(prompts),16):
        if (ROOT/'STOP_REQUESTED.json').exists():
            data.flush();mask.flush();atomic_json(out/'metadata.json',progress)
            status('stopped_by_request',resume_feature_row=progress['completed_rows']);return
        batch=prompts[offset:offset+16];values,masks=encoder.encode(batch,'cpu')
        assert values.dtype==torch.bfloat16 and values.shape[1:]==shape[1:]
        bits=values.contiguous().view(torch.uint16).numpy();end=offset+len(batch)
        data[offset:end]=bits;mask[offset:end]=masks.numpy()
        assert np.array_equal(data[offset:end],bits) and np.array_equal(mask[offset:end],masks.numpy())
        if end%256==0 or end==len(prompts):
            data.flush();mask.flush();progress['completed_rows']=end
            progress['text_identity']=encoder.identity
            atomic_json(out/'metadata.json',progress)
            elapsed=time.time()-begin
            status('precompute_text',done=end,total=len(prompts),bytes=progress['bytes'],
                   eta_seconds=elapsed/max(end-start,1)*(len(prompts)-end))
    progress['complete']=True;atomic_json(out/'metadata.json',progress)
    cached=DiskTextEncoder(out,encoder.identity)
    if start<len(prompts):
        x,y=cached.encode(batch,'cpu')
        assert torch.equal(values,x) and torch.equal(masks,y)
    atomic_json(out/'roundtrip_check.json',{'full_length_bfloat16_bitwise_equal':True,'mask_equal':True,
        'scope':'Every written batch checked before durable cursor advancement; final batch also checked through reader when written in this invocation.'})
    status('precompute_complete',prompts=len(prompts),bytes=progress['bytes'])


def train():
    from mcgen.training import Trainer
    stop = {'requested': False, 'trainer': None}
    def request_stop(signum, frame):
        stop['requested'] = True
        if stop['trainer'] is not None: stop['trainer'].stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    cfg=Experiment.load(ROOT/'config.yaml');wait_gpu(12)
    encoder=DiskTextEncoder(ROOT/'text_features',load_checkpoint(PARENT)['text_identity'])
    latest=ROOT/'training/latest.pt';kwargs={'initialize':PARENT}
    if latest.exists():
        saved=load_checkpoint(latest);step=saved['step'];kwargs={'resume':latest}
        metrics=ROOT/'training/metrics.jsonl'
        if metrics.exists():
            lines=metrics.read_text().splitlines();retained=[l for l in lines if json.loads(l)['step']<=step]
            if len(lines)!=len(retained):
                shutil.copyfile(metrics,metrics.with_name(f'metrics_before_resume_{int(time.time())}.jsonl'))
                metrics.write_text('\n'.join(retained)+'\n')
    elif (ROOT/'training').exists():
        # Preserve any interrupted startup before the first checkpoint.
        (ROOT/'training').rename(ROOT/f'training_interrupted_{int(time.time())}')
    trainer=Trainer(cfg,ROOT/'catalog',ROOT/'training',DEVICE,DEVICE,encoder=encoder,**kwargs)
    stop['trainer'] = trainer
    trainer.stop_requested = stop['requested']
    status('training',step=trainer.step,endpoint=cfg.train.steps,effective_batch=64)
    begin=time.time();trainer.run()
    if trainer.step < cfg.train.steps:
        status('stopped_by_request',step=trainer.step,checkpoint=str(latest))
        return
    atomic_json(ROOT/'training_complete.json',{'step':trainer.step,'seconds_this_invocation':time.time()-begin,
        'latest_sha256':file_hash(latest),'best_sha256':file_hash(ROOT/'training/best_flow.pt')})


def run():
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=json.load(open(ROOT/'protocol.json'))
        assert file_hash(ROOT/'config.yaml')==protocol['config_sha256']
        assert file_hash(PARENT)==protocol['initialization_sha256']
        try:
            if not (ROOT/'catalog/catalog.json').exists():annotate();build()
            command('features',['scripts/queue_fullscale_text.py','features','--device',DEVICE])
            command('training',['scripts/queue_fullscale_text.py','train','--device',DEVICE])
            for best in (False,True):
                name='evaluation_best_flow' if best else 'evaluation'
                if not (ROOT/(name+'_complete.json')).exists():
                    if (ROOT/name).exists():(ROOT/name).rename(ROOT/(name+'_interrupted_'+str(int(time.time()))))
                    command(name,['scripts/evaluate_text_recovery.py','--root',str(ROOT),'--device',DEVICE]+(['--best-flow'] if best else []))
                    atomic_json(ROOT/(name+'_complete.json'),{'completed_unix':time.time()})
            status('complete',step=200000,train_descriptions=10000,production_promoted=False)
        except Exception as e:
            status('failed',error=repr(e),resume_command=f'{sys.executable} scripts/queue_fullscale_text.py run --device {DEVICE}')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','features','train']);parser.add_argument('--device',default=DEVICE)
    args=parser.parse_args();DEVICE=args.device;torch.set_num_threads(4);globals()[args.stage]()
