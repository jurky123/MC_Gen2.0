"""Batch semantic-name extraction using fixed visual observations; no image inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mcgen.catalog import atomic_json, file_hash
from mcgen.semantic_captions import NAME_INSTRUCTION, name_messages, merge_candidate


def main(args):
    rows=[json.loads(x) for x in args.input.read_text().splitlines()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate input IDs')
    args.out.mkdir(parents=True,exist_ok=True)
    identity={'input_sha256':file_hash(args.input),'model':str(args.model.resolve()),
              'fusion_code_sha256':file_hash(Path(__file__).resolve().parents[1]/'src/mcgen/semantic_captions.py'),
              'instruction':NAME_INSTRUCTION,'max_new_tokens':256,'enable_thinking':False,
              'batch_size':args.batch_size,'torch':str(torch.__version__),
              'transformers':__import__('transformers').__version__,
              'files':{p.name:file_hash(p) for p in args.model.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt')}}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=args.out/'candidates.jsonl';existing=[]
    if path.exists():
        existing=[json.loads(x) for x in path.read_text().splitlines()]
        if any(r['signature']!=signature for r in existing):raise ValueError('Changed protocol; use a new output directory')
    done={r['id'] for r in existing};pending=[r for r in rows if r['id'] not in done]
    atomic_json(args.out/'protocol.json',dict(identity,signature=signature,visual_source='input observations; no vision inference in this stage'))
    valid=[]
    for r in pending:
        if not isinstance(r.get('visual_en'),str) or not r['visual_en'].strip():
            with path.open('a') as f:
                f.write(json.dumps({'id':r['id'],'target_hash':r['target_hash'],'signature':signature,
                    'status':'quarantine','quality_flags':['missing_visual_evidence'],
                    'approved_for_training':False})+'\n')
        else:valid.append(r)
    pending=valid
    if not pending:return
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(args.device).eval()
    for start in range(0,len(pending),args.batch_size):
        begin=time.time();chunk=pending[start:start+args.batch_size]
        chats=[name_messages(r['source_filename'],r['visual_en']) for r in chunk]
        prompts=[tokenizer.apply_chat_template(c,tokenize=False,add_generation_prompt=True,enable_thinking=False) for c in chats]
        inputs=tokenizer(prompts,return_tensors='pt',padding=True).to(args.device)
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=256,do_sample=False)
        for r,raw in zip(chunk,tokenizer.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)):
            try:result=merge_candidate(raw,r['source_filename'],r['visual_en'],r.get('visual_quality_flags',[]))
            except (ValueError,TypeError) as exc:
                result={'status':'quarantine','quality_flags':['schema_error'],'approved_for_training':False,'error':str(exc)}
            result.update(id=r['id'],target_hash=r['target_hash'],raw=raw,signature=signature)
            with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({'done':start+len(chunk),'total':len(pending),'seconds':time.time()-begin}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/iflab/models/Qwen3-8B'));p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.batch_size<1:p.error('batch size must be positive')
    torch.set_num_threads(4);main(args)
