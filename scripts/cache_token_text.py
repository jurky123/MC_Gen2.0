"""Resumable, disjoint chunk workers for exact singleton Qwen token features."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.qwen_text_tokens import QwenVLTextTokens
from prepare_token_full_training import DEFAULT


def read(path):return json.loads(path.read_text())


def arrays(root):
    cache=root/'text_features';p=read(cache/'plan.json');n=p['count']
    offsets=np.memmap(cache/'offsets.uint64.mmap',mode='r',dtype='uint64',shape=(n+1,))
    inp=np.memmap(cache/'input_offsets.uint64.mmap',mode='r',dtype='uint64',shape=(n+1,))
    tokens=np.memmap(cache/'input_ids.int32.mmap',mode='r',dtype='int32',shape=(int(inp[-1]),))
    positions=np.memmap(cache/'positions.uint16.mmap',mode='r',dtype='uint16',shape=(int(offsets[-1]),))
    features=np.memmap(cache/'features.uint16.mmap',mode='r+',dtype='uint16',shape=tuple(p['shape']))
    return cache,p,offsets,inp,tokens,positions,features


def worker(root,device):
    cfg=Experiment.load(root/'config.yaml')
    cache,p,offsets,inp,tokens,positions,features=arrays(root)
    for name,key in [('prompts.json','prompts_sha256'),('offsets.uint64.mmap','offsets_sha256'),
                     ('input_offsets.uint64.mmap','input_offsets_sha256'),('input_ids.int32.mmap','input_ids_sha256'),
                     ('positions.uint16.mmap','positions_sha256')]:
        if file_hash(cache/name)!=p[key]:raise ValueError('Token cache plan changed: '+name)
    encoder=QwenVLTextTokens(cfg.text,cfg.model,device)
    if encoder.identity!=p['text_identity']:raise ValueError('Actual Qwen identity differs from the cache plan')
    prompts=read(cache/'prompts.json')
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    completed=0;started=time.monotonic();chunk=p['chunk_prompts']
    while not stopping[0]:
        busy=False
        for lo in range(0,p['count'],chunk):
            hi=min(lo+chunk,p['count']);done=cache/'chunks'/f'{lo:07d}.json'
            if stopping[0]:return
            if done.exists():
                v=read(done)
                if (v['text_identity'],v['lo'],v['hi'])!=(p['text_identity'],lo,hi):
                    raise ValueError('Incompatible completed token chunk')
                continue
            with (cache/'locks'/f'{lo:07d}.lock').open('w') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    busy=True;continue
                if done.exists():continue
                begin=time.monotonic();canary=False
                for i in range(lo,hi):
                    a,b=int(offsets[i]),int(offsets[i+1]);u,v=int(inp[i]),int(inp[i+1])
                    row={'input_ids':tokens[u:v].tolist(),'positions':positions[a:b].tolist()}
                    values=encoder.encode_prepared(row,'cpu')
                    if values.dtype!=torch.bfloat16 or values.shape!=(b-a,4096):
                        raise ValueError('Cache must contain exact BF16 caption token states')
                    features[a:b]=values.view(torch.uint16).numpy()
                    if not canary and b>a:
                        check,mask=encoder.encode([prompts[i]],'cpu')
                        if int(mask.sum())!=b-a or not torch.equal(check[0,:b-a],values):
                            raise ValueError('Prepared token cache differs from live singleton inference')
                        canary=True
                a,b=int(offsets[lo]),int(offsets[hi])
                if b>a:features._mmap.flush(a*8192,(b-a)*8192)
                digest=hashlib.sha256(np.asarray(features[a:b]).tobytes()).hexdigest()
                atomic_json(done,{'lo':lo,'hi':hi,'token_start':a,'token_stop':b,
                    'features_sha256':digest,'text_identity':encoder.identity,
                    'prompts_sha256':p['prompts_sha256'],'live_inference_canary_exact':canary,
                    'seconds':time.monotonic()-begin,'device':device})
                completed+=hi-lo
                atomic_json(cache/f'progress_{device.replace(":","")}.json',{'done_this_worker':completed,
                    'elapsed_seconds':time.monotonic()-started,'last_lo':lo,'last_hi':hi,
                    'recent_prompts_per_second':(hi-lo)/(time.monotonic()-begin)})
                print('CACHED',device,lo,hi,'seconds',round(time.monotonic()-begin,2),flush=True)
        if not busy:return
        time.sleep(1)


def verify_chunks(root,prefix=None):
    cache,p,offsets,_,_,_,features=arrays(root)
    stop=p['count'] if prefix is None else prefix
    for lo in range(0,stop,p['chunk_prompts']):
        hi=min(lo+p['chunk_prompts'],p['count']);v=read(cache/'chunks'/f'{lo:07d}.json')
        a,b=int(offsets[lo]),int(offsets[hi])
        if (v['lo'],v['hi'],v['text_identity'],v['prompts_sha256'])!=(lo,hi,p['text_identity'],p['prompts_sha256']):
            raise ValueError('Invalid completed cache chunk')
        if hashlib.sha256(np.asarray(features[a:b]).tobytes()).hexdigest()!=v['features_sha256']:
            raise ValueError('Completed token cache chunk changed')
    return cache,p,offsets


def bind(root,features,catalog,recipe):
    meta=read(catalog/'catalog.json')
    spec={'catalog':str(catalog.resolve()),'catalog_fingerprint':meta['fingerprint'],
        'features':str(features.resolve()),'text_identity':read(features/'metadata.json')['text_identity'],
        'feature_hashes':{n:file_hash(features/n) for n in ('metadata.json','prompts.json','offsets.uint64.mmap')},
        'sampling':'uniform target then existing caption; full eligible data'}
    atomic_json(root/'text_replay.json',spec)
    recipe={**recipe,'text_replay_sha256':file_hash(root/'text_replay.json')}
    atomic_json(root/'recipe.json',recipe)


def probe(root):
    cache,p,offsets,_,_,_,_=arrays(root)
    n=p['probe_prompt_count']
    verify_chunks(root,n)
    features=root/'probe_features';features.mkdir(exist_ok=True)
    atomic_json(features/'prompts.json',read(cache/'prompts.json')[:n])
    np.asarray(offsets[:n+1]).tofile(features/'offsets.uint64.mmap')
    remaining=int(offsets[n])*8192
    with (cache/'features.uint16.mmap').open('rb') as source,(features/'features.uint16.mmap').open('wb') as dest:
        while remaining:
            block=source.read(min(16<<20,remaining))
            if not block:raise ValueError('Truncated cache prefix')
            dest.write(block);remaining-=len(block)
    atomic_json(features/'metadata.json',{**p,'count':n,'shape':[int(offsets[n]),4096],'complete':True,
        'features_bytes':int(offsets[n])*8192,'actual_max_tokens':int(np.diff(offsets[:n+1]).max()),
        'prompts_sha256':file_hash(features/'prompts.json'),
        'offsets_sha256':file_hash(features/'offsets.uint64.mmap'),
        'features_sha256':file_hash(features/'features.uint16.mmap')})
    out=root/'probe';out.mkdir(exist_ok=True)
    for name in ('config.yaml','pairs.json','review_snapshot.json','data_audit.json','excluded_pairs.json','initialization.json'):
        shutil.copy2(root/name,out/name)
    for name,target in [('text_features',features),('visual_features',root/'visual_features')]:
        if not (out/name).exists():(out/name).symlink_to(target.resolve(),target_is_directory=True)
    bind(out,features,root/'probe_catalog',read(root/'recipe.pending.json'))
    atomic_json(root/'probe_ready.json',{'complete':True,'prompts':n,'tokens':int(offsets[n]),'identity':p['text_identity']})
    print('PROBE CACHE READY',n,flush=True)


def finalize(root):
    cache,p,_=verify_chunks(root)
    atomic_json(cache/'metadata.json',{**p,'complete':True,
        'features_sha256':file_hash(cache/'features.uint16.mmap')})
    bind(root,cache,root/'full_text_catalog',read(root/'recipe.pending.json'))
    atomic_json(root/'cache_complete.json',{'complete':True,'prompts':p['count'],
        'tokens':p['shape'][0],'text_identity':p['text_identity'],'features_bytes':p['features_bytes']})
    print('FULL TOKEN CACHE COMPLETE',p['count'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('worker','probe','finalize'))
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='worker':
        torch.cuda.set_device(a.device);torch.cuda.set_per_process_memory_fraction(.45,a.device)
        worker(a.out,a.device)
    elif a.stage=='probe':probe(a.out)
    else:finalize(a.out)
