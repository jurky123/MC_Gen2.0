"""Resumable independent image observation followed by resource-name verification."""
import argparse,hashlib,io,json,time,zipfile,fcntl
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.images import content_hash,composite
from mcgen.grounded_labels import VERSION,VISUAL_INSTRUCTION,NAME_INSTRUCTION,parse_json,resource_flags,decide
ROOT=Path('outputs/grounded_label_revision_20261103')
SOURCE=Path('outputs/fullscale_text10k_20261103/catalog/samples.jsonl')
MODEL='/home/iflab/models/Qwen3-VL-8B-Instruct'
MAPS={}

def pixels(r):
 s=r['target'];key=s['path']
 if key not in MAPS:MAPS[key]=np.memmap(key,mode='r',dtype='uint8',shape=tuple(s['shape']))
 a=np.array(MAPS[key][s['index']]);assert content_hash(a)==r['target_hash'];return a

def prepare():
 ROOT.mkdir(exist_ok=False)
 archives=json.load(open('outputs/training_label_audit_20261103/archive_audit.json'));index=defaultdict(list);suffixes={}
 for url,archive in {r['url']:r['archive_file'] for r in archives}.items():
  with zipfile.ZipFile(archive) as z:
   for n in z.namelist():
    if n.endswith('emissive.properties'):
     for line in z.read(n).decode(errors='replace').splitlines():
      if line.strip().startswith('suffix.emissive='):suffixes[url]=line.split('=',1)[1].strip()
    if not n.lower().endswith('.png'):continue
    try:
     im=Image.open(io.BytesIO(z.read(n))).convert('RGBA')
     if im.size==(16,16):index[(url,content_hash(im))].append(n)
    except (OSError,ValueError):continue
 urls=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=['version_url'])['version_url'].to_pylist()
 oldlong={};train={}
 for line in SOURCE.open():
  r=json.loads(line)
  if r['split']!='train':continue
  if r['caption_variant']=='long':oldlong[r['target_hash']]=r['prompt']
  else:train[r['target_hash']]=r
 priority=[r['target_hash'] for r in json.load(open('outputs/training_label_audit_20261103/sample.json'))]
 order=list(dict.fromkeys(priority+sorted(train)))
 counts=Counter()
 with (ROOT/'cases.jsonl').open('w') as out:
  for h in order:
   r=train[h];url=urls[r['source_row']];paths=[p for p in index[(url,h)] if Path(p).name==r['source_filename']];visible=int(np.count_nonzero(pixels(r)[...,3]>8))
   flags=resource_flags(r['source_filename'],paths,suffixes.get(url))
   case={**r,'old_filename_label':r['prompt'],'old_visual_label':oldlong.get(h),'resource_paths':paths,'resource_url':url,'path_status':'matched_original_visible_rgba' if paths else 'unrecovered','resource_flags':flags,'visible_pixels':visible}
   out.write(json.dumps(case,ensure_ascii=False)+'\n');counts['path_resolved' if paths else 'path_unresolved']+=1
   if flags or visible<8:counts['pre_quarantine']+=1
 signature={'version':VERSION,'cases_sha256':file_hash(ROOT/'cases.jsonl'),'policy_sha256':file_hash('src/mcgen/grounded_labels.py'),'annotator_sha256':file_hash(__file__),'model_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b','model_verified_manifest_sha256':file_hash('outputs/text_caption_scale1152_20261102/vision/download/verified.json'),'visual_instruction':VISUAL_INSTRUCTION,'name_instruction':NAME_INSTRUCTION,'visual_sees_filename':False,'visual_sees_old_labels':False,'visual_views':'white and gray nearest 256','max_tokens':[192,192],'test_and_val_used':False,'scope':'All current train targets, beginning with 72 audited examples. No training restart.','counts':dict(counts),'total':len(order)}
 signature['signature']=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();atomic_json(ROOT/'protocol.json',signature);print(signature['counts'],flush=True)

def annotate(args):
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 protocol=json.load(open(ROOT/'protocol.json'));assert protocol['policy_sha256']==file_hash('src/mcgen/grounded_labels.py');assert protocol['annotator_sha256']==file_hash(__file__)
 signature=protocol['signature'];rows=[json.loads(s) for s in (ROOT/'cases.jsonl').open()];done=set()
 for path in ROOT.glob('labels_*.jsonl'):
  for line in path.open():
   r=json.loads(line);assert r['signature']==signature;done.add(r['target_hash'])
 if args.limit:rows=rows[:args.limit]
 rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard and r['target_hash'] not in done]
 labelpath=ROOT/f'labels_{args.tag}.jsonl';lock=open(ROOT/f'{args.tag}.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42,args.device)
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);processor.tokenizer.padding_side='left'
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(args.device).eval()
 def generate(messages):
  inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True}).to(args.device)
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=192,do_sample=False)
  return processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 started=time.time();count=0;decisions=Counter()
 for start in range(0,len(rows),args.batch_size):
  chunk=rows[start:start+args.batch_size];active=[];results=[]
  for r in chunk:
   if r['resource_flags'] or r['visible_pixels']<8:
    results.append({**r,'file_label':None,'visual_label':None,'caption_variants':[],'decision':'quarantine','quality_flags':r['resource_flags']+(['insufficient_visible_content'] if r['visible_pixels']<8 else []),'visual_observation':None,'note':'Provenance/content rejection before annotation; no invented whole-object labels.'})
   else:active.append(r)
  messages=[]
  for r in active:
   a=pixels(r);views=[Image.fromarray(composite(a,bg).round().astype('uint8')).resize((256,256),Image.Resampling.NEAREST) for bg in [255,128]]
   messages.append([{'role':'user','content':[{'type':'image','image':v} for v in views]+[{'type':'text','text':VISUAL_INSTRUCTION}]}])
  rawvisual=generate(messages) if active else [];visuals=[];valid=[];name_messages=[]
  for r,raw in zip(active,rawvisual):
   try:v=parse_json(raw,['object','role','caption','certainty'])
   except (ValueError,TypeError) as e:
    results.append({**r,'file_label':None,'visual_label':None,'caption_variants':[],'decision':'quarantine','quality_flags':['visual_schema_error'],'error':str(e),'raw_visual':raw});continue
   visuals.append((r,v,raw));valid.append(r)
   payload={'filename':r['source_filename'],'paths':r['resource_paths'],'path_status':r['path_status'],'independent_visual':v}
   name_messages.append([{'role':'user','content':[{'type':'text','text':NAME_INSTRUCTION+'\nDATA:\n'+json.dumps(payload)}]}])
  rawname=generate(name_messages) if valid else []
  for (r,v,rawv),rawn in zip(visuals,rawname):
   try:s=parse_json(rawn,['semantic_name','removed_tokens','relation','reason']);decision=decide(r,v,s)
   except (ValueError,TypeError) as e:decision={'file_label':None,'visual_label':None,'caption_variants':[],'decision':'quarantine','quality_flags':['review_schema_error'],'error':str(e),'visual_observation':v}
   results.append({**r,**decision,'raw_visual':rawv,'raw_name':rawn})
  with labelpath.open('a') as out:
   for r in results:
    r.update(signature=signature,label_version=VERSION);out.write(json.dumps(r,ensure_ascii=False)+'\n');decisions[r['decision']]+=1
  count+=len(chunk);atomic_json(ROOT/f'status_{args.tag}.json',{'stage':'running','done_this_run':count,'pending_at_start':len(rows),'seconds':time.time()-started,'decisions':dict(decisions)})
  print(count,len(rows),dict(decisions),round(time.time()-started,1),flush=True)
 atomic_json(ROOT/f'status_{args.tag}.json',{'stage':'complete','done_this_run':count,'seconds':time.time()-started,'decisions':dict(decisions)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','annotate']);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--tag',default='pilot');args=p.parse_args()
 if args.stage=='prepare':prepare()
 else:annotate(args)
