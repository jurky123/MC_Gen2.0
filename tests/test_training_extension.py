import copy
from collections import OrderedDict
from types import SimpleNamespace
import pytest
import torch
from mcgen.conditioning import TextEncoder
from mcgen.config import TrainConfig
from mcgen.smoke import fixture,tiny_config
from mcgen.training import Trainer


def test_trim_keeps_all_valid_positions_including_left_padding():
    enc=TextEncoder.__new__(TextEncoder)
    enc.cfg=SimpleNamespace(backend='hf');enc.model_cfg=SimpleNamespace(text_length=8)
    enc.cache_limit=100000;enc.cache=OrderedDict();enc.hits=enc.misses=enc.cache_bytes=0
    features=torch.randn(8,12);mask=torch.tensor([0,0,1,1,1,0,0,0],dtype=torch.bool)
    enc.cache['x']=(features,mask);enc.trim_padding=False
    full,m=enc.encode(['x','x'],'cpu');enc.trim_padding=True
    trimmed,tm=enc.encode(['x','x'],'cpu')
    assert trimmed.shape==(2,5,12)
    assert torch.equal(trimmed,full[:,:5]) and torch.equal(tm,m[:,:5])
    tm.zero_();assert enc.cache['x'][1].sum()==3


def test_schedule_offset_and_optimizer_continuation(tmp_path):
    torch.set_num_threads(1)
    path=fixture(tmp_path/'catalog');cfg=tiny_config();cfg.train.steps=4;cfg.train.warmup=1
    first=Trainer(cfg,path,tmp_path/'first');first.run(stop_after=2)
    state=torch.load(tmp_path/'first/latest.pt',weights_only=False)
    ext=copy.deepcopy(cfg);ext.train.steps=6;ext.train.schedule_start=2;ext.train.warmup=2
    later=Trainer(ext,path,tmp_path/'later',initialize=tmp_path/'first/latest.pt')
    later.optimizer.load_state_dict(state['optimizer']);later.step=state['step'];later.run(stop_after=3)
    assert later.step==3
    assert later.optimizer.param_groups[0]['lr']==pytest.approx(ext.train.lr/2)
    assert all(int(s['step'])==3 for s in later.optimizer.state.values() if 'step' in s)
    resumed=Trainer(ext,path,tmp_path/'later',resume=tmp_path/'later/latest.pt');resumed.run()
    assert resumed.step==6


def test_schedule_offset_validation():
    with pytest.raises(ValueError):TrainConfig(steps=10,schedule_start=10,warmup=0)
    with pytest.raises(ValueError):TrainConfig(steps=10,schedule_start=8,warmup=3)


def test_trimmed_padding_preserves_model_output_and_gradients():
    from mcgen.model import PixelFlow
    cfg=tiny_config().model;cfg.byte_tokens=False;cfg.text_normalization='layernorm'
    torch.manual_seed(42);model=PixelFlow(cfg)
    # Activate residual gates/output: zero-initialized networks would make this vacuous.
    for parameter in model.parameters():
        if parameter.requires_grad:parameter.data.normal_(0,.1)
    features=torch.randn(2,8,cfg.text_dim);mask=torch.zeros(2,8,dtype=torch.bool);mask[:,:5]=True
    ref=torch.randn(2,4,cfg.ref_size,cfg.ref_size);present=torch.zeros(2,dtype=torch.bool)
    x=torch.randn(2,4,cfg.size,cfg.size);t=torch.tensor([.2,.7]);results=[];gradients=[]
    for length in [8,5]:
        model.zero_grad(set_to_none=True)
        out=model(x,t,model.context(features[:,:length],mask[:,:length],ref,present))
        out.square().mean().backward();results.append(out.detach());gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(results[0],results[1],atol=1e-6,rtol=1e-5)
    assert gradients[0].keys()==gradients[1].keys()
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=1e-6,rtol=1e-4)
