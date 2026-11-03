"""Precompute official pooled embeddings; publish real probe data and verify protocol."""
import hashlib,importlib.util,json,time,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.images import content_hash,encode
from mcgen.tasks import conditioned_prompts
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')

def verify_official(encoder):
 source=ROOT/'official_embedding_reference.py';spec=importlib.util.spec_from_file_location('official_embedding_reference',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 impl=encoder.embedding
 class Adapter:
  device=impl.device
  def __call__(self,**inputs):return SimpleNamespace(last_hidden_state=impl.model(**inputs,use_cache=False).last_hidden_state,attention_mask=inputs['attention_mask'])
 reference=object.__new__(module.Qwen3VLEmbedder);reference.model=Adapter();reference.processor=impl.processor;reference.default_instruction="Represent the user's input.";reference.max_length=512
 reference.min_pixels=4096;reference.max_pixels=1843200;reference.total_pixels=7864320;reference.fps=1;reference.max_frames=64
 prompts=['Minecraft pixel art; item: a red sword','Minecraft pixel art; item: soul gem','','Minecraft pixel art; item: copper pickaxe with a dark wooden handle']
 with torch.inference_mode():
  got,_=encoder.encode(prompts,impl.device);expected=reference.process([{'text':p} if p else {} for p in prompts])
 cosine=torch.nn.functional.cosine_similarity(got[:,0].float(),expected.float());error=(got[:,0].float()-expected.float()).abs().max().item()
 assert cosine.min().item()>.99999,(cosine,error)
 atomic_json(ROOT/'embedding_protocol_check.json',{'official_source_sha256':file_hash(source),'minimum_cosine':cosine.min().item(),'max_absolute_difference':error,'official_final_token_pooling_and_normalization':True,'embedding_identity':encoder.identity})
 print('official protocol verified',error,flush=True)

def run(device):
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,device);cfg=Experiment.load(ROOT/'config.yaml');encoder=TextEncoder(cfg.text,cfg.model,device);verify_official(encoder)
 # Publish real native-grid training examples before the full cache, so tuning can run concurrently.
 examples=json.load(open(ROOT/'sample.json'));images=[];maps={}
 for r in examples:
  s=r['target'];key=s['path']
  if key not in maps:maps[key]=np.memmap(key,mode='r',dtype='uint8',shape=tuple(s['shape']))
  a=np.array(maps[key][s['index']]);assert content_hash(a)==r['target_hash'];images.append(encode(a))
 text,mask=encoder.encode(conditioned_prompts([r['prompt'] for r in examples],cfg),'cpu')
 temp=ROOT/'probe_data.tmp';torch.save({'image':torch.stack(images),'text':text,'mask':mask},temp);temp.replace(ROOT/'probe_data.pt');print('probe data ready',flush=True)
 prompts=[]
 for line in (ROOT/'catalog/samples.jsonl').open():prompts.append(json.loads(line)['prompt'])
 prompts += [r['evaluation_prompt'] for r in json.load(open(ROOT/'eval_cases.json'))]+['']
 prompts=sorted(set(conditioned_prompts(prompts,cfg)));out=ROOT/'features';out.mkdir(exist_ok=True);shape=[len(prompts),1,4096]
 if (out/'metadata.json').exists():
  meta=json.load(open(out/'metadata.json'));assert meta['text_identity']==encoder.identity
  if meta['complete']:return
 if not (out/'prompts.json').exists():atomic_json(out/'prompts.json',prompts)
 assert json.load(open(out/'prompts.json'))==prompts
 atomic_json(out/'metadata.json',{'complete':False,'shape':shape,'text_identity':encoder.identity,'prompts_sha256':file_hash(out/'prompts.json')})
 mode='r+' if (out/'features.uint16.mmap').exists() else 'w+'
 f=np.memmap(out/'features.uint16.mmap',mode=mode,dtype='uint16',shape=tuple(shape));m=np.memmap(out/'masks.uint8.mmap',mode=mode,dtype='uint8',shape=tuple(shape[:2]));start=json.load(open(out/'progress.json'))['done'] if (out/'progress.json').exists() else 0;begin=time.time()
 for i in range(start,len(prompts),128):
  h,mask=encoder.encode(prompts[i:i+128],'cpu');f[i:i+len(h)]=h.view(torch.uint16).numpy();m[i:i+len(h)]=mask.numpy()
  if (i-start)%1024==0 or i+len(h)==len(prompts):
   f.flush();m.flush();atomic_json(out/'progress.json',{'done':i+len(h),'total':len(prompts),'seconds_this_run':time.time()-begin});print('cached',i+len(h),len(prompts),round(time.time()-begin,1),flush=True)
 for i in [0,len(prompts)//2//128*128,(len(prompts)-1)//128*128]:
  h,mask=encoder.encode(prompts[i:i+128],'cpu');assert np.array_equal(f[i:i+len(h)],h.view(torch.uint16).numpy());assert np.array_equal(m[i:i+len(h)],mask.numpy())
 atomic_json(out/'metadata.json',{'complete':True,'shape':shape,'text_identity':encoder.identity,'prompts_sha256':file_hash(out/'prompts.json')});print('CACHE COMPLETE',flush=True)
if __name__=='__main__':run('cuda:1')
