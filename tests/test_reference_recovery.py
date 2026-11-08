import copy

import torch

from mcgen.model import PixelFlow
from mcgen.reference_recovery import (JOINT_MIX, WARM_MIX, RecoveryData,RecoverySampler,coarse_caption,
                                     learning_rates,phase_for,trainable_phase)
from mcgen.smoke import tiny_config


def test_reference_joint_labels_drop_design_specific_names_and_detail():
    rows=[{'quality_tier':q,'split':s,'target_hash':str(i),'captions':['Gold mage helmet','Long observed design']}
          for i,(q,s) in enumerate([('highest_quality','train'),('reviewed','train'),
                                   ('highest_quality','val'),('highest_quality','test')])]
    sampler=RecoverySampler(rows,JOINT_MIX)
    ids,tasks,variants=sampler.sample(8192,torch.Generator().manual_seed(18))
    assert set(ids)=={0,1}
    assert all(v in (0,2) for t,v in zip(tasks,variants) if t=='joint')
    joint=[v for t,v in zip(tasks,variants) if t=='joint']
    assert .65<joint.count(2)/len(joint)<.75
    assert all(rows[i]['quality_tier']=='highest_quality' for i,t in zip(ids,tasks) if t in ('text','unconditional'))
    assert any(v==1 for t,v in zip(tasks,variants) if t=='text')
    _,warm_tasks,_=RecoverySampler(rows,WARM_MIX).sample(1024,torch.Generator().manual_seed(9))
    assert set(warm_tasks)=={'joint','reference'}


def test_coarse_conditions_are_nonempty_and_preserve_category():
    assert coarse_caption('Gold mage samurai helmet')=='helmet'
    assert coarse_caption('Dreadfire battleaxe')=='axe'
    assert coarse_caption('Iron pickaxe')=='pickaxe'
    assert coarse_caption('Valkyrie crown charm')=='charm'
    assert coarse_caption('Silk bow tie')=='item'
    assert coarse_caption('a strange object')=='item'


def test_update_budget_has_long_constant_lr_and_a_protected_visual_phase():
    recipe={'visual_warmup_steps':2000,'steps':8000,'decay_start':6000,
            'warmup_steps':200,'base_lr':1e-6,'visual_lr':1e-4}
    assert phase_for(1999,recipe)=='visual' and phase_for(2000,recipe)=='joint'
    assert learning_rates(1000,recipe)==(0.,1e-4)
    assert learning_rates(1999,recipe)==(0.,1e-4)
    assert learning_rates(2199,recipe)==(1e-6,1e-4)
    assert learning_rates(5999,recipe)==(1e-6,1e-4)
    base,visual=learning_rates(7999,recipe)
    assert base>=.3e-6 and visual>=.3e-4


def test_visual_warmup_freezes_every_backbone_parameter():
    cfg=copy.deepcopy(tiny_config().model)
    cfg.visual_adapter=True;cfg.visual_dim=12;cfg.visual_grid=2;cfg.visual_layers=[0,1]
    model=PixelFlow(cfg)
    trainable_phase(model,'visual')
    assert all(p.requires_grad==name.startswith('visual_adapter.') for name,p in model.named_parameters())
    trainable_phase(model,'joint')
    assert all(p.requires_grad for p in model.parameters())


def test_cached_batch_keeps_images_text_variants_and_presence_aligned():
    import numpy as np
    data=RecoveryData.__new__(RecoveryData)
    data.rows=[{}, {}, {}]
    data.targets=torch.randn(3,4,16,16)
    data.text=torch.randn(3,2,1,8).bfloat16()
    data.mask=torch.ones(3,2,1,dtype=torch.bool)
    data.coarse_text=torch.randn(3,1,8).bfloat16()
    data.coarse_mask=torch.ones(3,1,dtype=torch.bool)
    data.features=torch.randn(3,4,12).bfloat16().view(torch.uint16).numpy().copy()
    data.cached=None
    batch=data.batch([2,0,1],['joint','reference','text'],[2,0,1],'cpu')
    assert torch.equal(batch[1][0],data.coarse_text[2])
    assert torch.equal(batch[1][2],data.text[1,1])
    assert batch[2][:,0].tolist()==[True,False,True]
    assert batch[4].tolist()==[True,True,False]
    data.cache_device('cpu')
    cached=data.batch([2,0,1],['joint','reference','text'],[2,0,1],'cpu')
    assert all(torch.equal(a,b) for a,b in zip(batch,cached))
