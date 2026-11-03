"""Official Qwen3-VL-Embedding text protocol: final attended token + L2 normalization.

Protocol reference: QwenLM/Qwen3-VL-Embedding src/models/qwen3_vl_embedding.py.
No visual labels, intermediate-layer concatenation, or image input is used here.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import unicodedata
import torch
import torch.nn.functional as F
from .catalog import file_hash
from .config import fingerprint

MAX_INPUT_LENGTH=512
DEFAULT_INSTRUCTION="Represent the user's input."

def conversations(prompts,instruction=''):
 instruction=instruction.strip() or DEFAULT_INSTRUCTION
 if not unicodedata.category(instruction[-1]).startswith('P'):instruction+='.'
 return [[{'role':'system','content':[{'type':'text','text':instruction}]},
          {'role':'user','content':[{'type':'text','text':p if p else 'NULL'}]}] for p in prompts]

def last_attended(hidden,mask):
 if not mask.bool().any(-1).all():raise ValueError('Embedding input has no attended token')
 positions=mask.shape[1]-1-mask.flip(-1).long().argmax(-1)
 return hidden[torch.arange(len(hidden),device=hidden.device),positions]

class QwenVLEmbedding:
 def __init__(self,cfg,model_cfg,device):
  from transformers import AutoModel,AutoProcessor
  if model_cfg.text_dim!=4096 or model_cfg.text_length!=1:raise ValueError('Qwen VL embedding requires one4096-dimensional token')
  # Only relocate physical files. The saved configuration and all verified
  # artifact hashes still determine identity, preserving checkpoint/cache checks.
  root=Path(os.environ.get('MCGEN_EMBEDDING_MODEL') or cfg.model)
  manifest=json.load(open(root/'verified_download.json'))
  if not manifest['complete'] or manifest['repo']!='Qwen/Qwen3-VL-Embedding-8B' or manifest['revision']!=cfg.revision:raise ValueError('Wrong or unverified embedding model')
  actual={}
  for row in manifest['files']:
   digest=file_hash(root/row['name'])
   if digest!=row['verified_sha256']:raise ValueError('Embedding artifacts changed: '+row['name'])
   actual[row['name']]=digest
  info={'config':{k:v for k,v in asdict(cfg).items() if k!='cache_mb'},'protocol':'qwen3_vl_final_attended_l2_v1','max_input_length':MAX_INPUT_LENGTH,'output_shape':[1,4096],'files':actual}
  self.identity=fingerprint(info);self.device=device;self.cfg=cfg;self.hits=0
  self.processor=AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=False,padding_side='right')
  self.model,loading=AutoModel.from_pretrained(root,local_files_only=True,trust_remote_code=False,dtype=getattr(torch,cfg.dtype),attn_implementation='sdpa',output_loading_info=True)
  if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):raise ValueError('Embedding weight mapping mismatch: '+str(loading))
  self.model=self.model.to(device).eval();self.model.requires_grad_(False)
 @torch.inference_mode()
 def encode(self,prompts,output_device):
  if not prompts:raise ValueError('empty prompt batch')
  text=self.processor.apply_chat_template(conversations(prompts,self.cfg.instruction),add_generation_prompt=True,tokenize=False)
  inputs=self.processor(text=text,padding=True,truncation=False,do_resize=False,return_tensors='pt').to(self.device)
  if inputs['input_ids'].shape[1]>MAX_INPUT_LENGTH:raise ValueError('Embedding input exceeds512 tokens; truncation is forbidden')
  outputs=self.model(**inputs,use_cache=False)
  values=F.normalize(last_attended(outputs.last_hidden_state,inputs['attention_mask']),p=2,dim=-1)
  if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite embedding')
  self.hits+=len(prompts)
  return values[:,None].to(output_device),torch.tensor([[bool(p.strip())] for p in prompts],dtype=torch.bool,device=output_device)
 def cache_stats(self):return {'backend':'official_qwen3_vl_embedding','hits':self.hits,'misses':0}
