import json
import numpy as np
import pytest
import torch
from mcgen.catalog import atomic_json, file_hash
from mcgen.disk_text import DiskTextEncoder


def make_cache(root):
    values=torch.randn(3,8,12).bfloat16()
    masks=torch.tensor([[True]*8,[True]*3+[False]*5,[False]*8])
    values.view(torch.uint16).numpy().tofile(root/'features.uint16.mmap')
    masks.numpy().astype('uint8').tofile(root/'masks.uint8.mmap')
    atomic_json(root/'prompts.json',['one','two',''])
    atomic_json(root/'metadata.json',{'complete':True,'shape':[3,8,12],
        'text_identity':'frozen','prompts_sha256':file_hash(root/'prompts.json')})
    return values,masks


def test_exact_bfloat16_and_independent_masks(tmp_path):
    values,masks=make_cache(tmp_path);encoder=DiskTextEncoder(tmp_path,'frozen')
    x,m=encoder.encode(['two','','one','two'],'cpu')
    assert torch.equal(x,values[[1,2,0,1]]) and torch.equal(m,masks[[1,2,0,1]])
    m[:]=False;x[:]=0
    a,b=encoder.encode(['one'],'cpu')
    assert torch.equal(a,values[:1]) and torch.equal(b,masks[:1])
    with pytest.raises(KeyError):encoder.encode(['missing'],'cpu')


def test_fail_closed_identity_index_and_completion(tmp_path):
    make_cache(tmp_path)
    with pytest.raises(ValueError,match='identity'):DiskTextEncoder(tmp_path,'other')
    meta=json.loads((tmp_path/'metadata.json').read_text());meta['complete']=False
    atomic_json(tmp_path/'metadata.json',meta)
    with pytest.raises(ValueError,match='Incomplete'):DiskTextEncoder(tmp_path)
    meta['complete']=True;atomic_json(tmp_path/'metadata.json',meta)
    atomic_json(tmp_path/'prompts.json',['changed','two',''])
    with pytest.raises(ValueError,match='index'):DiskTextEncoder(tmp_path)

def test_merged_cache_routes_exact_features_in_original_order(tmp_path):
    from mcgen.disk_text import MergedDiskTextEncoder
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    av,am=make_cache(a);bv,bm=make_cache(b)
    atomic_json(b/'prompts.json',['three','four',''])
    meta=json.loads((b/'metadata.json').read_text());meta['prompts_sha256']=file_hash(b/'prompts.json');atomic_json(b/'metadata.json',meta)
    encoder=MergedDiskTextEncoder([DiskTextEncoder(a),DiskTextEncoder(b)])
    x,m=encoder.encode(['four','one','','three','four'],'cpu')
    assert torch.equal(x,torch.stack([bv[1],av[0],av[2],bv[0],bv[1]]))
    assert torch.equal(m,torch.stack([bm[1],am[0],am[2],bm[0],bm[1]]))
    with pytest.raises(KeyError):encoder.encode(['not cached'],'cpu')
    meta['text_identity']='other';atomic_json(b/'metadata.json',meta)
    with pytest.raises(ValueError,match='Incompatible'):MergedDiskTextEncoder([DiskTextEncoder(a),DiskTextEncoder(b)])
