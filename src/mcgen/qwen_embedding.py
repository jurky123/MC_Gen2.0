"""Official Qwen3-VL-Embedding protocol: final attended token + L2 normalization.

Protocol reference: QwenLM/Qwen3-VL-Embedding src/models/qwen3_vl_embedding.py.
Text and multimodal inputs share the frozen model; their protocols are identified separately.
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
IMAGE_MIN_PIXELS=4096
IMAGE_MAX_PIXELS=512*512

def embedding_image(value):
 from PIL import Image,ImageOps
 import numpy as np
 if isinstance(value,np.ndarray):
  if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[-1] not in (3,4):raise ValueError('reference must be uint8 RGB/RGBA')
  value=Image.fromarray(value)
 if not isinstance(value,Image.Image):raise ValueError('reference must be a PIL image or uint8 RGB/RGBA array')
 value=ImageOps.exif_transpose(value).convert('RGBA')
 return Image.alpha_composite(Image.new('RGBA',value.size,'white'),value).convert('RGB')

def multimodal_conversations(prompts,images,instruction=''):
 if not prompts or len(prompts)!=len(images) or any(im is None for im in images):raise ValueError('one image required per multimodal input')
 result=conversations(prompts,instruction)
 for conversation,p,im in zip(result,prompts,images):
  conversation[1]['content']=[{'type':'image','image':embedding_image(im),'min_pixels':IMAGE_MIN_PIXELS,'max_pixels':IMAGE_MAX_PIXELS}]
  if p.strip():conversation[1]['content'].append({'type':'text','text':p})
 return result

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
  self.multimodal_identity=fingerprint({'text_identity':self.identity,'protocol':'qwen3_vl_image_then_text_final_attended_l2_v2','min_pixels':IMAGE_MIN_PIXELS,'max_pixels':IMAGE_MAX_PIXELS,'alpha_background':'white','exif_transpose':True,'batch_policy':'single_sample'})
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
 @torch.inference_mode()
 def encode_multimodal(self,prompts,images,output_device):
  if not prompts or len(prompts)!=len(images) or any(im is None for im in images):raise ValueError('one image required per multimodal input')
  # BF16 VL features can shift with batch shape/padding. Canonical single-sample
  # forwards keep offline training caches identical to interactive inference.
  chunks=[self._encode_multimodal_batch([p],[im],output_device) for p,im in zip(prompts,images)]
  return torch.cat([v[0] for v in chunks]),torch.cat([v[1] for v in chunks])
 @torch.inference_mode()
 def _encode_multimodal_batch(self,prompts,images,output_device):
  from qwen_vl_utils import process_vision_info
  messages=multimodal_conversations(prompts,images,self.cfg.instruction)
  text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  # Match the official Qwen preprocessing. Fail on invalid images; never silently
  # fall back to a NULL text embedding or truncate away image tokens.
  vision,_,kwargs=process_vision_info(messages,image_patch_size=16,return_video_metadata=True,return_video_kwargs=True)
  inputs=self.processor(text=text,images=vision,padding=True,truncation=False,do_resize=False,return_tensors='pt',**kwargs).to(self.device)
  if inputs['input_ids'].shape[1]>MAX_INPUT_LENGTH:raise ValueError('Embedding input exceeds512 tokens; truncation is forbidden')
  outputs=self.model(**inputs,use_cache=False)
  values=F.normalize(last_attended(outputs.last_hidden_state,inputs['attention_mask']),p=2,dim=-1)
  if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite image embedding')
  self.hits+=len(prompts)
  return values[:,None].to(output_device),torch.ones((len(prompts),1),dtype=torch.bool,device=output_device)
 def cache_stats(self):return {'backend':'official_qwen3_vl_embedding','hits':self.hits,'misses':0}
