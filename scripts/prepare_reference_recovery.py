"""Reuse the full lossless16 data with a longer, reference-heavy training recipe."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from mcgen.catalog import atomic_json,file_hash
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.reference_recovery import JOINT_MIX,WARM_MIX,RecoveryData,coarse_caption
from mcgen.runtime import load_checkpoint
from mcgen.tasks import conditioned_prompts

COMMON=['diamond sword','iron pickaxe','gold ingot','red gem','diamond helmet','iron axe','emerald','shield']


def prepare(out,device):
    source=Path('outputs/reference_native16_lossless_20261107').resolve()
    out=out.resolve()
    if out.exists():raise ValueError('Choose a new experiment directory')
    out.mkdir(parents=True)
    for name in ('pairs.json','review_snapshot.json','data_audit.json','excluded_pairs.json'):
        shutil.copy2(source/name,out/name)
    rows=json.loads((out/'pairs.json').read_text())
    audit=json.loads((out/'data_audit.json').read_text())
    assert audit['lossy_target_count']==0 and len(rows)==2470
    assert file_hash(out/'pairs.json')==audit['pairs_sha256']
    (out/'visual_features').symlink_to(source/'visual_features',target_is_directory=True)
    cfg=Experiment.load(source/'config.yaml')
    cfg.train.steps=8000;cfg.train.batch_size=128;cfg.train.warmup=200
    cfg.train.lr=1e-6;cfg.train.val_samples=48;cfg.train.eval_every=1000;cfg.train.task_mix=JOINT_MIX
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
    checkpoint=source/'training/best_joint_flow.pt'
    state=load_checkpoint(checkpoint)
    if state.get('diagnostic_only') or state.get('target_leakage'):raise ValueError('Never initialize production from a fitting control')
    assert state['step']==270 and state['config']['model']['size']==16
    identity=state['text_identity'];visual_identity=state['visual_identity']
    digest=hashlib.sha256()
    for name,tensor in sorted(state['model'].items()):
        if not name.startswith(('image_adapter.','visual_adapter.')):
            digest.update(name.encode());digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    recipe={'version':'reference_recovery_v1','steps':8000,'batch_size':128,'visual_warmup_steps':2000,
        'base_lr':1e-6,'visual_lr':1e-4,'warmup_steps':200,'decay_start':6000,
        'lr_floor':.3,'warm_task_mix':WARM_MIX,'joint_task_mix':JOINT_MIX,
        'reference_source_hq_fraction':.6,'joint_text':{'coarse_category':.7,'original_short':.3},
        'pure_text':{'hq_only':True,'original_detail':.7,'original_short':.3},
        'evaluate_updates':[0,500,1000,2000,3000,4000,6000,8000],
        'primary_sampling_cfg':1.,'secondary_sampling_cfg':2.,'solver':'heun','sampling_steps':24,
        'reference_best_metric':'mean high-noise matched joint/coarse and image-only Flow MSE at t=.9,.99',
        'source_pairs_sha256':audit['pairs_sha256'],'source_step':270,'data_changed':False,
        'oracle_features_or_overfit_weights_used':False}
    atomic_json(out/'recipe.json',recipe)
    atomic_json(out/'initialization.json',{'checkpoint':str(checkpoint),'sha256':file_hash(checkpoint),
        'source_step':270,'source_run':str(source),'transfer':'native16 full model copied exactly; optimizer reset',
        'generator_lr':1e-6,'visual_adapter_lr':1e-4,'frozen_encoders':True,
        'initial_backbone_sha256':digest.hexdigest(),'initial_text_baseline':'previous full-data270 model; not identical to original4128 after its joint fine-tuning',
        'budget_kind':'fixed_optimizer_updates','planned_steps':8000,'batch_size':128,
        'overfit_or_oracle_checkpoint_used':False})
    del state;gc.collect()
    old=DiskTextEncoder(source/'text_features',expected_identity=identity)
    base=DiskTextEncoder('outputs/qwen_vl_embedding_large_20261103/features',expected_identity=identity)
    categories=[coarse_caption(r['captions'][0]) for r in rows]
    old_prompts=json.loads((source/'text_features/prompts.json').read_text())
    prompts=list(dict.fromkeys(old_prompts+conditioned_prompts(sorted(set(categories))+COMMON,cfg)))
    missing=[p for p in prompts if p not in old.index and p not in base.index]
    atomic_json(out/'cache_progress.json',{'phase':'text_setup','new_prompt_count':len(missing),
        'category_count':len(set(categories)),'encoders_frozen':True})
    text_dir=out/'text_features';text_dir.mkdir()
    atomic_json(text_dir/'prompts.json',prompts)
    shape=[len(prompts),1,cfg.model.text_dim]
    features=np.memmap(text_dir/'features.uint16.mmap',mode='w+',dtype='uint16',shape=tuple(shape))
    masks=np.memmap(text_dir/'masks.uint8.mmap',mode='w+',dtype='uint8',shape=tuple(shape[:2]))
    encoder=None
    if missing:
        torch.set_num_threads(4)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.45,device)
        print('ENCODING NEW COARSE PROMPTS',missing,flush=True)
        encoder=TextEncoder(cfg.text,cfg.model,device)
        if encoder.identity!=identity:raise ValueError('Frozen text encoder identity changed')
    for i,prompt in enumerate(prompts):
        provider=old if prompt in old.index else base if prompt in base.index else encoder
        h,m=provider.encode([prompt],'cpu')
        if not bool(torch.isfinite(h).all()):raise ValueError('Nonfinite conditioning features')
        features[i]=h[0].contiguous().view(torch.uint16).numpy()
        masks[i]=m[0].numpy().astype('uint8')
    features.flush();masks.flush()
    for i,p in enumerate(old_prompts):
        assert np.array_equal(features[i],old.features[old.index[p]])
        assert np.array_equal(masks[i],old.masks[old.index[p]])
    atomic_json(text_dir/'metadata.json',{'complete':True,'shape':shape,'done':len(prompts),
        'text_identity':identity,'prompts_sha256':file_hash(text_dir/'prompts.json'),
        'features_sha256':file_hash(text_dir/'features.uint16.mmap'),'masks_sha256':file_hash(text_dir/'masks.uint8.mmap'),
        'original_features_copied_exactly':True,'new_prompt_policy':'singleton canonical frozen encoder; coarse category only'})
    del encoder,features,masks;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    atomic_json(out/'conditioning_map.json',[{'id':r['id'],'source_caption':r['captions'][0],
        'coarse_condition':category,'split':r['split']} for r,category in zip(rows,categories)])
    data=RecoveryData(out)
    assert len(data.rows)==2470 and data.visual_identity==visual_identity
    atomic_json(out/'cache_progress.json',{'phase':'complete','new_prompt_count':len(missing),
        'category_count':len(set(categories)),'original_text_features_preserved':True})
    atomic_json(out/'preparation.json',{'complete':True,'source_pairs_sha256':audit['pairs_sha256'],
        'same_image_pairs_captions_and_splits':file_hash(source/'pairs.json')==file_hash(out/'pairs.json'),
        'training_pairs':1064,'training_unique_targets':864,'validation_pairs':1350,'test_pairs':56,
        'lossy_target_count':0,'source_step':270,'native_resolution':16,'new_text_prompts':missing})
    print('PREPARED8000 updates, batch128; all1064 train pairs /864 targets retained.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_recovery_20261108'))
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();prepare(args.out,args.device)
