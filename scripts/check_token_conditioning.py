"""Disposable real-target test of the token condition and frozen time path."""
import argparse
import json
from pathlib import Path

import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.five_task_reference import FiveTaskData,configure_trainable,accumulate_loss,noisy_batch
from train_reference_recovery import initialize,optimizer
from prepare_token_full_training import DEFAULT


def check(root,device):
    data=FiveTaskData(root/'probe');data.cache_device(device)
    model,info=initialize(data,device);configure_trainable(model,True)
    times={k:v.detach().clone() for k,v in model.time.state_dict().items()}
    ids=[];seen=set()
    for i,r in enumerate(data.rows):
        if r['split']=='train' and r['quality_tier']=='highest_quality' and data.categories[i] not in seen:
            ids.append(i);seen.add(data.categories[i])
            if len(ids)==8:break
    if len(ids)!=8:raise ValueError('Eight distinct HQ target categories required')
    batch=data.batch(ids,['text']*8,[1]*8,device)
    target,h,mask,_,_=batch
    h=h.detach().float().requires_grad_(True)
    pure=(target,h,mask,None,None)
    noisy=noisy_batch(target,2026110941,device)
    accumulate_loss(model,pure,noisy,8,sort_by_length=True)
    valid_grad=((h.grad.abs().sum(-1)>0)&mask).sum(-1)
    padding_erased=not bool(h.grad[~mask].any())
    text_grad=model.text.weight.grad.float().norm().item()
    no_visual=all(p.grad is None for p in model.visual_adapter.parameters())
    model.zero_grad(set_to_none=True)
    fixed=[noisy_batch(target,seed,device) for seed in (2026110943,2026110947)]
    def score():
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            ctx=model.context(batch[1],mask,None,None)
            return [float((model(x,t,ctx).float()-v).square().mean()) for x,t,v in fixed]
    before=score()
    opt=optimizer(model,{'base_lr':1e-5,'visual_lr':0.,'text_lr':1e-4})
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        accumulate_loss(model,(target,batch[1],mask,None,None),
            noisy_batch(target,2026110951+step,device),8,sort_by_length=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    after=score()
    exact=all(torch.equal(times[k],v) for k,v in model.time.state_dict().items())
    unchanged=file_hash(info['checkpoint'])==info['sha256']
    passed=(bool((valid_grad>1).all()) and padding_erased and text_grad>0 and no_visual
        and exact and unchanged and sum(after)<sum(before))
    report={'passed':passed,'targets':8,'caption_valid_tokens':mask.sum(-1).tolist(),
        'tokens_with_nonzero_input_gradient':valid_grad.tolist(),
        'padding_has_zero_gradient':padding_erased,'text_projection_gradient_norm':text_grad,
        'pure_text_has_no_visual_gradient':no_visual,'before_fresh_noise_loss':before,
        'after_fresh_noise_loss':after,'temporary_updates':150,
        'time_mlp_bitwise_unchanged':exact,'source_checkpoint_unchanged':unchanged,
        'temporary_weights_discarded':True,'check_only_learning_rates':{'base':1e-5,'text':1e-4},
        'purpose':'Small real-target learnability, not held-out quality or final training recipe'}
    atomic_json(root/'token_condition_check.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not passed:raise RuntimeError('Token conditioning check failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--device',default='cuda:1');a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.cuda.set_per_process_memory_fraction(.3,a.device)
    check(a.out,a.device)
