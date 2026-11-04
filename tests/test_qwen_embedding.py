import pytest,torch
from dataclasses import replace
from mcgen.qwen_embedding import conversations,last_attended,QwenVLEmbedding,multimodal_conversations,embedding_image
from mcgen.smoke import tiny_config
from mcgen.model import PixelFlow

def test_pooling_handles_both_padding_sides():
 x=torch.arange(2*4*3).reshape(2,4,3)
 assert torch.equal(last_attended(x,torch.tensor([[1,1,0,0],[0,0,1,1]])),torch.stack([x[0,1],x[1,3]]))
 with pytest.raises(ValueError):last_attended(x,torch.zeros(2,4))
def test_official_text_template_contains_no_image_or_old_prompt():
 c=conversations(['sun gem','']);assert c[0][0]['content'][0]['text']=="Represent the user's input."
 assert c[0][1]['content']==[{'type':'text','text':'sun gem'}]
 assert c[1][1]['content'][0]['text']=='NULL'

def test_image_only_has_no_null_caption_and_alpha_is_composited():
 import numpy as np
 from PIL import Image
 a=Image.new('RGBA',(32,32),(200,0,0,0))
 messages=multimodal_conversations(['','red sword'],[a,a])
 assert [v['type'] for v in messages[0][1]['content']]==['image']
 assert [v['type'] for v in messages[1][1]['content']]==['image','text']
 assert messages[1][1]['content'][1]['text']=='red sword'
 assert np.array(embedding_image(a))[0,0].tolist()==[255,255,255]
 with pytest.raises(ValueError):multimodal_conversations([''],[None])
 with pytest.raises(ValueError):multimodal_conversations([''],[])
 with pytest.raises(ValueError):embedding_image(np.zeros((4,4,4),dtype=float))

def test_image_encoding_uses_identical_singleton_forwards_for_any_request_batch():
 from types import MethodType
 model=object.__new__(QwenVLEmbedding);calls=[]
 def encode(self,prompts,images,device):
  calls.append((prompts,images));return torch.full((1,1,8),float(images[0])),torch.ones(1,1,dtype=torch.bool)
 model._encode_multimodal_batch=MethodType(encode,model)
 values,mask=model.encode_multimodal(['a','b'],[1,2],'cpu')
 assert calls==[(['a'],[1]),(['b'],[2])]
 assert values[:,0,0].tolist()==[1.,2.] and mask.all()

def test_explicit_multimodal_transfer_uses_pooled_condition_not_spatial_branch(monkeypatch):
 from types import SimpleNamespace
 from mcgen.runtime import Runtime
 import mcgen.runtime as module
 cfg=tiny_config();cfg.task='text';cfg.model.conditioning='text_only';cfg.text.backend='qwen3_vl_embedding'
 runtime=object.__new__(Runtime);runtime.config=cfg;runtime.device='cpu';runtime.model=object()
 calls=[]
 def encode(prompts,images,device):
  calls.append((prompts,images));return torch.ones(1,1,8),torch.ones(1,1,dtype=torch.bool)
 runtime.encoder=SimpleNamespace(encode_multimodal=encode)
 def sampler(model,text,mask,**kwargs):
  assert kwargs['text_scale']==3.;assert 'reference' not in kwargs
  assert kwargs['seeds']==[9];assert mask.all()
  return torch.zeros(1,4,8,8)
 monkeypatch.setattr(module,'sample',sampler)
 refs=[object()]
 runtime.generate([''],refs,seeds=[9],conditioning_mode='multimodal_embedding',ref_scale=3.,reference_mode=None)
 assert calls==[([''],refs)]
 with pytest.raises(ValueError,match='text-only'):runtime.generate([''],refs)
 with pytest.raises(ValueError,match='every sample'):runtime.generate([''],conditioning_mode='multimodal_embedding')
 with pytest.raises(ValueError,match='reference_mode'):runtime.generate([''],refs,conditioning_mode='multimodal_embedding',reference_mode='shape')
 runtime.generate(['red sword'],refs,seeds=[9],conditioning_mode='multimodal_embedding',ref_scale=3.)
 assert calls[-1][0][0].endswith(': red sword')
def test_text_only_architecture_does_not_encode_dummy_reference():
 cfg=replace(tiny_config().model,conditioning='text_only',byte_tokens=False,text_dim=8,text_length=1)
 m=PixelFlow(cfg);assert not hasattr(m,'reference_encoder')
 text=torch.randn(2,1,8);mask=torch.ones(2,1,dtype=torch.bool)
 ctx=m.context(text,mask,None,None);assert ctx[0].shape==(2,2,cfg.width)
 y=m(torch.randn(2,4,cfg.size,cfg.size),torch.rand(2),ctx);assert y.shape==(2,4,cfg.size,cfg.size)
 with pytest.raises(ValueError):m.context(text,mask,torch.zeros(2,4,cfg.ref_size,cfg.ref_size),torch.ones(2,dtype=torch.bool))


def test_embedding_relocation_preserves_identity_and_verifies_files(tmp_path, monkeypatch):
 import hashlib,json,shutil
 import transformers
 from mcgen.config import TextConfig
 original=tmp_path/'original';original.mkdir()
 payload=b'original frozen embedding weights'
 (original/'model.safetensors').write_bytes(payload)
 manifest={'complete':True,'repo':'Qwen/Qwen3-VL-Embedding-8B','revision':'a'*40,
           'files':[{'name':'model.safetensors','verified_sha256':hashlib.sha256(payload).hexdigest()}]}
 (original/'verified_download.json').write_text(json.dumps(manifest))
 relocated=tmp_path/'relocated';shutil.copytree(original,relocated)
 loaded=[]
 class FrozenModel:
  def to(self,*args):return self
  def eval(self):return self
  def requires_grad_(self,*args):return self
 def load_model(path,**kwargs):
  loaded.append(path)
  return FrozenModel(),{}
 monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *a,**k: object())
 monkeypatch.setattr(transformers.AutoModel,'from_pretrained',load_model)
 monkeypatch.delenv('MCGEN_EMBEDDING_MODEL',raising=False)
 text=TextConfig(backend='qwen3_vl_embedding',model=str(original),revision='a'*40)
 model=replace(tiny_config().model,text_dim=4096,text_length=1)
 first=QwenVLEmbedding(text,model,'cpu')
 monkeypatch.setenv('MCGEN_EMBEDDING_MODEL',str(relocated))
 second=QwenVLEmbedding(text,model,'cpu')
 assert second.identity==first.identity
 assert loaded==[original,relocated]
 assert text.model==str(original)
 (relocated/'model.safetensors').write_bytes(b'changed weights')
 with pytest.raises(ValueError,match='artifacts changed'):
  QwenVLEmbedding(text,model,'cpu')
