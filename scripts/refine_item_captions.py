"""Recover valid observations, then reconcile filenames while looking at the actual images."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import audit_item_captions as audit
from mcgen.captions import parse_candidate
from mcgen.caption_review import REVIEW_INSTRUCTION,parse_review
from mcgen.catalog import file_hash,atomic_json
from mcgen.semantic_captions import natural_name,filename_words
from mcgen.images import composite

ROOT=Path(__file__).resolve().parents[1]

def main(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True);(out/'semantic').mkdir(exist_ok=True)
    cases=[json.loads(s) for s in (args.source/'cases.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in (args.source/'candidates.jsonl').read_text().splitlines()]
    if (out/'cases.jsonl').exists():assert file_hash(out/'cases.jsonl')==file_hash(args.source/'cases.jsonl')
    else:shutil.copyfile(args.source/'cases.jsonl',out/'cases.jsonl')
    recovered=[]
    for r in old:
        try:result=parse_candidate(r['raw'])
        except (ValueError,TypeError) as exc:
            result={'annotation':None,'quality_flags':['schema_error'],'status':'quarantine',
                    'approved_for_training':False,'error':str(exc)}
        base={k:v for k,v in r.items() if k!='error'}
        if r.get('error'):base['prior_parser_error']=r['error']
        recovered.append({**base,**result,'visual_en':(result['annotation'] or {}).get('short_caption'),
            'visual_quality_flags':result['quality_flags'],'recovery_source':'same raw response, soft formatting limits'})
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in recovered))
    lookup={r['id']:r for r in recovered}
    identity={'cases_sha256':file_hash(out/'cases.jsonl'),'visual_source_sha256':file_hash(args.source/'candidates.jsonl'),
        'instruction':REVIEW_INSTRUCTION,'model_verified_manifest_sha256':file_hash(args.source/'download/verified.json'),
        'code':{f:file_hash(ROOT/f) for f in ['src/mcgen/caption_review.py','src/mcgen/semantic_captions.py','src/mcgen/captions.py']},
        'max_new_tokens':384,'batch_size':args.batch_size,'image_views':'single neutral gray background, nearest 256',
        'prior_caption_exposed_to_reviewer':False,
        'torch':str(torch.__version__),'transformers':__import__('transformers').__version__}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    atomic_json(out/'protocol.json',dict(identity,signature=signature))
    path=out/'semantic/candidates.jsonl';existing=[]
    if path.exists():
        existing=[json.loads(s) for s in path.read_text().splitlines()]
        if any(r['signature']!=signature for r in existing):raise ValueError('Changed review protocol; use new output')
    done={r['id'] for r in existing};rows=[r for r in cases if r['id'] not in done]
    if args.limit:rows=rows[:args.limit]
    if not rows:return
    device=torch.device(args.device)
    free,total=torch.cuda.mem_get_info(device)
    if free<23*1024**3:raise RuntimeError(f'Insufficient free GPU memory for safe shared inference: {free}')
    torch.cuda.set_per_process_memory_fraction(min(0.30,(free-3*1024**3)/total),device)
    processor=AutoProcessor.from_pretrained(audit.MODEL,local_files_only=True,trust_remote_code=False)
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(audit.MODEL,dtype=torch.bfloat16,
        local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to(device).eval()
    for start in range(0,len(rows),args.batch_size):
        chunk=rows[start:start+args.batch_size];messages=[];begin=time.time()
        for r in chunk:
            pic=Image.fromarray(np.uint8(np.rint(composite(audit.array(r),128)))).resize((256,256),Image.Resampling.NEAREST)
            content=[{'type':'image','image':pic}]
            context={'filename':r['source_filename']}
            messages.append([{'role':'user','content':content+[{'type':'text','text':REVIEW_INSTRUCTION+'\nINPUT DATA:\n'+json.dumps(context,ensure_ascii=False)}]}])
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',processor_kwargs={'padding':True}).to(device)
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=384,do_sample=False)
        for r,raw in zip(chunk,processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)):
            try:result=parse_review(raw,r['source_filename'])
            except (ValueError,TypeError) as exc:
                visual=lookup[r['id']]['visual_en'] or '';name=' '.join(filename_words(r['source_filename']))
                result={'status':'quarantine','quality_flags':['review_schema_error'],'error':str(exc),
                        'effective_name':name,'visual_text':visual,'combined_text':name+'. '+visual if visual else name,
                        'proposed_views':[],'approved_for_training':False,'fallback':'retained independent observation, review failed'}
            result.update(id=r['id'],target_hash=r['target_hash'],source_filename=r['source_filename'],
                raw=raw,signature=signature,independent_visual=lookup[r['id']]['annotation'])
            with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({'done':start+len(chunk),'total':len(rows),'seconds':round(time.time()-begin,2),
                          'peak_reserved_gib':round(torch.cuda.max_memory_reserved(device)/1024**3,2)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4);main(args)
