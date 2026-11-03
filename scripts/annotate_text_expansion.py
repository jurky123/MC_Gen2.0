"""Reuse the accepted image+filename reviewer on new targets; explicit weak labels."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from mcgen.catalog import atomic_json,file_hash
from mcgen.caption_review import REVIEW_INSTRUCTION,parse_review
from mcgen.images import composite
from audit_item_captions import array,MODEL

ROOT=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--limit',type=int,default=0);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();ROOT=a.root;torch.set_num_threads(4)
 rows=[json.loads(s) for s in (ROOT/'annotation_cases.jsonl').read_text().splitlines()]
 identity={'cases_sha256':file_hash(ROOT/'annotation_cases.jsonl'),'instruction':REVIEW_INSTRUCTION,'parser_sha256':file_hash('src/mcgen/caption_review.py'),'semantic_parser_sha256':file_hash('src/mcgen/semantic_captions.py'),'image_view':'single neutral-gray background nearest 256','model_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b','verified_manifest_sha256':file_hash('outputs/text_caption_scale1152_20261102/vision/download/verified.json'),'max_new_tokens':384,'batch_size':16,'independent_image_only_observation':False,'prior_caption_exposed':False,'note':'Same final image+filename caption instruction as accepted pipeline. The separate image-only observation, which was never passed to the reviewer, is not repeated for this expansion.'}
 sig=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=ROOT/'annotations.jsonl'
 existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
 if any(r['signature']!=sig for r in existing):raise ValueError('Protocol changed; do not mix labels')
 atomic_json(ROOT/'annotation_protocol.json',{**identity,'signature':sig})
 done={r['id'] for r in existing};pending=[r for r in rows if r['id'] not in done]
 if a.limit:pending=pending[:a.limit]
 if not pending:raise SystemExit('No pending annotations')
 while torch.cuda.mem_get_info(a.device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(a.device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),a.device)
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);processor.tokenizer.padding_side='left'
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to(a.device).eval()
 for start in range(0,len(pending),16):
  batch=pending[start:start+16];messages=[];begin=time.time()
  for r in batch:
   im=Image.fromarray(composite(array(r),128).round().astype('uint8')).resize((256,256),Image.Resampling.NEAREST)
   messages.append([{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':REVIEW_INSTRUCTION+'\nINPUT DATA:\n'+json.dumps({'filename':r['source_filename']},ensure_ascii=False)}]}])
  inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True}).to(a.device)
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=384,do_sample=False)
  texts=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
  with path.open('a') as f:
   for r,raw in zip(batch,texts):
    try:result=parse_review(raw,r['source_filename'])
    except (ValueError,TypeError) as e:result={'combined_text':None,'quality_flags':['review_schema_error'],'approved_for_training':False,'status':'quarantine','error':str(e)}
    f.write(json.dumps({**result,'id':r['id'],'target_hash':r['target_hash'],'source_filename':r['source_filename'],'signature':sig,'raw':raw,'independent_visual':None},ensure_ascii=False)+'\n')
  status={'stage':'annotating','done':len(existing)+start+len(batch),'total':len(rows),'batch_seconds':round(time.time()-begin,2),'peak_reserved_gib':round(torch.cuda.max_memory_reserved(a.device)/1024**3,2)}
  atomic_json(ROOT/'status.json',status);print(json.dumps(status),flush=True)

 completed=len(existing)+len(pending)
 atomic_json(ROOT/'status.json',{'stage':'annotation_complete' if completed==len(rows) else 'annotation_partial','done':completed,'total':len(rows),'annotations_sha256':file_hash(path)})
