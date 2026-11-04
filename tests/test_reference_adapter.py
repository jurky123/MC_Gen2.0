"""Verify adaptation can learn while preserving the entire original text path."""
from dataclasses import replace
import pytest
import torch

from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.reference_adapter import initialize_adapter,backbone_digest
from mcgen.sampling import sample
from mcgen.smoke import tiny_config


def configuration():
    base=tiny_config()
    base.model=replace(base.model,conditioning='text_only',byte_tokens=False,text_dim=8,text_length=1)
    base.text.backend='hf';base.text.model='fake';base.text.revision='a'*40
    base.task='text';base.train.task_mix={'text':.9,'reference':0.,'joint':0.,'unconditional':.1}
    data=base.to_dict();data['model']['multimodal_adapter']=True
    data['text']['backend']='qwen3_vl_embedding';data['text']['model']='fake';data['text']['revision']='a'*40
    data['conditioning_input']='multimodal_embedding';data['task']='joint'
    data['train']['trainable']='embedding_adapter'
    data['train']['task_mix']={'text':0.,'reference':.5,'joint':.5,'unconditional':0.}
    return base,Experiment.from_dict(data)


def test_adapter_updates_and_text_and_null_remain_exact():
    torch.set_num_threads(1);torch.manual_seed(41)
    base,cfg=configuration();source=PixelFlow(base.model)
    # Smoke models initialize the output to zero; give this test a meaningful
    # pretrained condition response rather than comparing two zero predictors.
    with torch.no_grad():
        source.output.weight.normal_(std=.1)
        for b in source.blocks:b.modulation[-1].bias.fill_(.3)
    m=initialize_adapter(cfg,{'config':base.to_dict(),'model':source.state_dict()},'cpu')
    before=backbone_digest(m);h=torch.randn(2,1,8);mask=torch.ones(2,1,dtype=torch.bool);present=torch.ones(2,dtype=torch.bool)
    x=torch.randn(2,4,8,8);t=torch.rand(2)
    original=m(x,t,m.context(h,mask,None,None)).detach().clone()
    null=m(x,t,m.context(h,torch.zeros_like(mask),None,None)).detach().clone()
    with torch.no_grad():
        assert torch.equal(original,m(x,t,m.context(h,mask,None,present)))
    opt=torch.optim.AdamW(m.image_adapter.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();pred=m(x,t,m.context(h,mask,None,present));pred.square().mean().backward();opt.step()
    assert before==backbone_digest(m)
    assert torch.equal(original,m(x,t,m.context(h,mask,None,None)))
    assert torch.equal(null,m(x,t,m.context(h,torch.zeros_like(mask),None,present)))
    assert not torch.equal(original,m(x,t,m.context(h,mask,None,present)))
    assert all(not p.requires_grad and p.grad is None for n,p in m.named_parameters() if not n.startswith('image_adapter.'))
    # CFG null must never activate the new path, including image guidance=0.
    a=sample(m,h,mask,seeds=[1,2],steps=2,text_scale=0,embedding_adapter=True)
    b=sample(m,h,mask,seeds=[1,2],steps=2,text_scale=0)
    assert torch.equal(a,b)


def test_adapter_contract_rejects_unprotected_architecture():
    _,cfg=configuration();data=cfg.to_dict();data['train']['trainable']='all'
    with pytest.raises(ValueError,match='frozen-backbone'):Experiment.from_dict(data)
    data=cfg.to_dict();data['model']['byte_tokens']=True
    with pytest.raises(ValueError,match='without byte'):Experiment.from_dict(data)
