import pytest,torch
from dataclasses import replace
from mcgen.qwen_embedding import conversations,last_attended,QwenVLEmbedding
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
