import copy
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.smoke import tiny_config


def test_historical_config_preserves_raw_projection():
    cfg=tiny_config().to_dict();cfg['model'].pop('text_normalization')
    assert Experiment.from_dict(cfg).model.text_normalization=='none'
    cfg['model']['text_normalization']='typo'
    with pytest.raises(ValueError,match='text_normalization'):
        Experiment.from_dict(cfg)


def test_normalization_matches_diagnostic_hook_and_controls_token_scale():
    torch.set_num_threads(1)
    cfg=tiny_config().model
    raw=PixelFlow(cfg)
    normalized_cfg=copy.deepcopy(cfg);normalized_cfg.text_normalization='layernorm'
    normalized=PixelFlow(normalized_cfg);normalized.load_state_dict(raw.state_dict())
    features=torch.randn(2,5,cfg.text_dim)
    features[:,0]*=120
    mask=torch.ones(2,5,dtype=torch.bool)
    seen=[]
    hook=normalized.text.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
    actual=normalized.context(features,mask,None,None)
    hook.remove()
    assert torch.allclose(seen[0].square().mean(-1),torch.ones(2,5),atol=1e-4)
    handle=raw.text.register_forward_pre_hook(lambda module,args:(F.layer_norm(args[0],(args[0].shape[-1],)),))
    expected=raw.context(features,mask,None,None);handle.remove()
    for a,e in zip(actual[0],expected[0]):assert torch.equal(a,e)
    for a,e in zip(actual[1],expected[1]):assert torch.equal(a,e)
    # No prompt signal can leak through a null condition, even with outliers.
    null=torch.zeros_like(mask)
    a=normalized.context(features,null,None,None)
    b=normalized.context(features*100,null,None,None)
    for x,y in zip(a[0],b[0]):assert torch.equal(x,y)


def test_resume_accepts_historical_missing_default_field(tmp_path):
    from mcgen.training import Trainer
    from mcgen.smoke import fixture
    torch.set_num_threads(1)
    cfg=tiny_config();catalog=fixture(tmp_path/'fixture')
    trainer=Trainer(cfg,catalog,tmp_path/'train');checkpoint=trainer.run(stop_after=1)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state['config']['model'].pop('text_normalization')
    torch.save(state,checkpoint)
    resumed=Trainer(cfg,catalog,tmp_path/'train',resume=checkpoint)
    assert resumed.step==1
    assert resumed.cfg.model.text_normalization=='none'
    resumed.run(stop_after=2)
    changed=copy.deepcopy(cfg);changed.model.text_normalization='layernorm'
    with pytest.raises(ValueError,match='exactly matching'):
        Trainer(changed,catalog,tmp_path/'other',resume=checkpoint)
