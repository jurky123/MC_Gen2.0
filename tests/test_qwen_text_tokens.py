from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.disk_text import load_text_cache,RaggedDiskTextEncoder
from mcgen.qwen_text_tokens import tokenize_captions,identity_from_parent,QwenVLTextTokens


class Processor:
    def __init__(self):self.tokenizer=self;self.all_special_ids=[0]
    def apply_chat_template(self,messages,**kwargs):
        return ['SYSTEM: Represent input. USER:'+m[1]['content'][0]['text']+' ASSISTANT:' for m in messages]
    def __call__(self,texts,**kwargs):
        return {'input_ids':[[ord(c) for c in text] for text in texts],
                'offset_mapping':[[(i,i+1) for i in range(len(text))] for text in texts]}


def test_caption_mask_excludes_template_even_if_caption_repeats_system_words():
    processor=Processor();prompts=['Represent input. blue gem','银色头盔，蓝色宝石。','']
    rows=tokenize_captions(processor,prompts)
    assert rows[-1]=={'input_ids':[],'positions':[]}
    for p,row in zip(prompts,rows):
        assert ''.join(chr(row['input_ids'][i]) for i in row['positions'])==p
    with pytest.raises(ValueError,match='limit'):tokenize_captions(processor,['blue helmet'],max_tokens=2)
    with pytest.raises(ValueError,match='truncation'):tokenize_captions(processor,['x'*512])


def test_token_identity_cannot_alias_pooled_or_another_length():
    assert identity_from_parent('pooled',512)!='pooled'
    assert identity_from_parent('pooled',512)!=identity_from_parent('pooled',128)


def cache(root):
    import numpy as np
    root.mkdir()
    h=torch.randn(5,4096).bfloat16()
    h.view(torch.uint16).numpy().tofile(root/'features.uint16.mmap')
    np.asarray([0,3,5,5],dtype='uint64').tofile(root/'offsets.uint64.mmap')
    atomic_json(root/'prompts.json',['red sword','blue gem',''])
    atomic_json(root/'metadata.json',{'complete':True,'format':'ragged_bfloat16_text_v1',
        'shape':[5,4096],'max_tokens':512,'text_identity':'tokens',
        'prompts_sha256':file_hash(root/'prompts.json'),'offsets_sha256':file_hash(root/'offsets.uint64.mmap'),
        'features_sha256':file_hash(root/'features.uint16.mmap')})
    return h


def test_ragged_cache_preserves_every_token_and_pads_only_batch_without_quantizing(tmp_path):
    root=tmp_path/'cache';h=cache(root);encoder=load_text_cache(root,'tokens')
    assert isinstance(encoder,RaggedDiskTextEncoder)
    values,masks=encoder.encode(['blue gem','','red sword'],'cpu')
    assert values.shape==(3,3,4096)
    assert torch.equal(values[0,:2],h[3:]) and torch.equal(values[2],h[:3])
    assert masks.tolist()==[[True,True,False],[False,False,False],[True,True,True]]
    assert not values[1].any()
    values.zero_();again,_=encoder.encode(['red sword'],'cpu');assert torch.equal(again[0],h[:3])
    with pytest.raises(KeyError):encoder.encode(['missing'],'cpu')
    with pytest.raises(ValueError,match='identity'):load_text_cache(root,'pooled')


def test_cache_rejects_corruption_and_incomplete_features(tmp_path):
    root=tmp_path/'cache';cache(root)
    m=json.loads((root/'metadata.json').read_text());m['complete']=False;atomic_json(root/'metadata.json',m)
    with pytest.raises(ValueError,match='Incomplete'):load_text_cache(root)
    m['complete']=True;atomic_json(root/'metadata.json',m)
    with (root/'features.uint16.mmap').open('r+b') as f:
        first=f.read(1);f.seek(0);f.write(bytes([first[0]^1]))
    with pytest.raises(ValueError,match='changed'):load_text_cache(root)


def test_token_encoder_uses_all_content_states_and_singleton_forward():
    encoder=QwenVLTextTokens.__new__(QwenVLTextTokens)
    encoder.cfg=SimpleNamespace(dtype='bfloat16',instruction='')
    encoder.device='cpu';encoder.max_tokens=512;encoder.hits=0;encoder.processor=Processor()
    calls=[]
    def model(**kwargs):
        ids=kwargs['input_ids'];calls.append(ids.shape)
        return SimpleNamespace(last_hidden_state=ids.bfloat16()[...,None].repeat(1,1,4096))
    encoder.model=model
    values,masks=encoder.encode(['abc','xy',''],'cpu')
    assert len(calls)==2 and all(s[0]==1 for s in calls)
    assert masks.tolist()==[[True,True,True],[True,True,False],[False,False,False]]
    assert values[0,:,0].tolist()==[ord(c) for c in 'abc']
    assert values[1,:2,0].tolist()==[ord(c) for c in 'xy']
    with pytest.raises(ValueError,match='separate'):encoder.encode_multimodal([],[],None)
