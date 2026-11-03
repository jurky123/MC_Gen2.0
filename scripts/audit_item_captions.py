"""Image-only candidate captions, with names withheld until the review export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

from mcgen.captions import CAPTION_INSTRUCTION,parse_candidate
from mcgen.catalog import file_hash,atomic_json
from mcgen.images import native_grid,content_hash,composite

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/text_caption_audit_20261102'
MODEL=ROOT.parent/'Qwen3-VL-8B-Instruct'
MAPS={}


def array(row):
    spec=row['target'];key=(spec['path'],tuple(spec['shape']))
    if key not in MAPS:MAPS[key]=np.memmap(key[0],mode='r',dtype=np.uint8,shape=key[1])
    a=native_grid(np.array(MAPS[key][spec['index']]),16)
    assert content_hash(a)==row['target_hash']
    return a


def messages(row):
    a=array(row)
    views=[Image.fromarray(np.uint8(np.rint(composite(a,bg)))).resize((256,256),Image.Resampling.NEAREST) for bg in (255,32)]
    return [{'role':'user','content':[{'type':'image','image':v} for v in views]+[{'type':'text','text':CAPTION_INSTRUCTION}]}]


def annotate(device,batch_size,limit):
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    protocol=json.loads((OUT/'protocol.json').read_text());assert file_hash(OUT/'cases.jsonl')==protocol['cases_sha256']
    signature=hashlib.sha256(json.dumps({'revision':protocol['annotator_revision'],
        'parser_sha256':file_hash(ROOT/'src/mcgen/captions.py'),
        'instruction':CAPTION_INSTRUCTION,'cases':protocol['cases_sha256'],
        'max_new_tokens':220},sort_keys=True).encode()).hexdigest()
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
    if limit:rows=rows[:limit]
    path=OUT/'candidates.jsonl';done=set()
    if path.exists():
        existing=[json.loads(x) for x in path.read_text().splitlines()]
        if any(r.get('annotation_signature')!=signature for r in existing):
            raise ValueError('Annotation protocol changed; use a separate audit output instead of mixing candidates.')
        done={r['id'] for r in existing}
    rows=[r for r in rows if r['id'] not in done]
    if not rows:
        print('No pending cases.',flush=True)
        return
    index=MODEL/'model.safetensors.index.json'
    if not index.exists():
        raise FileNotFoundError(f'Visual annotator download is incomplete: {index}')
    shards=set(json.loads(index.read_text())['weight_map'].values())
    missing=[name for name in shards if not (MODEL/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Visual annotator shards missing: {missing}')
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to(device).eval()
    identity={p.name:file_hash(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.jinja')}
    atomic_json(OUT/'annotator_identity.json',{'repo':protocol['annotator'],'revision':protocol['annotator_revision'],
        'files':identity,'instruction':CAPTION_INSTRUCTION,'max_new_tokens':220,'do_sample':False,'batch_size':batch_size,
        'transformers_version':__import__('transformers').__version__,'torch_version':str(torch.__version__)})
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size];begin=time.time()
        inputs=processor.apply_chat_template([messages(r) for r in chunk],tokenize=True,padding=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt').to(device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=220,do_sample=False)
        raw=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for r,text in zip(chunk,raw):
            try:result=parse_candidate(text)
            except (ValueError,TypeError,json.JSONDecodeError) as exc:
                result={'annotation':None,'status':'quarantine','quality_flags':['schema_error'],
                        'approved_for_training':False,'error':str(exc)}
            result.update(id=r['id'],target_hash=r['target_hash'],raw=text,
                          annotation_signature=signature,batch_size=batch_size,
                          source_filename=r['source_filename'],
                          visual_en=(result['annotation'] or {}).get('short_caption'),
                          visual_quality_flags=result['quality_flags'])
            with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({'done':start+len(chunk),'pending_total':len(rows),'seconds':time.time()-begin,
                          'peak_reserved_gib':round(torch.cuda.max_memory_reserved(device)/1024**3,2)}),flush=True)


def review():
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
    candidate_path=OUT/'candidates.jsonl'
    candidates={r['id']:r for r in map(json.loads,candidate_path.read_text().splitlines())} if candidate_path.exists() else {}
    selected=rows
    pending={'annotation':None,'status':'not_annotated'}
    pages=OUT/'review';pages.mkdir(exist_ok=True)
    statuses={}
    # Generated export is separate from the reviewer's decisions, which must survive reruns.
    with (OUT/'review_candidates.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','target_hash','split','filename','old_caption','prior_visual_caption','new_short','new_detailed','auto_status','reviewer','decision','corrected_short','corrected_detailed','notes'])
        for r in selected:
            c=candidates.get(r['id'],pending);a=c['annotation'] or {};statuses[c['status']]=statuses.get(c['status'],0)+1
            writer.writerow([r['id'],r['target_hash'],r['split'],r['source_filename'],r['old_caption'],r['previous_descriptive'],a.get('short_caption',''),a.get('detailed_caption',''),c['status'],'','','','',''])
    for start in range(0,len(selected),8):
        sheet=Image.new('RGB',(1300,8*148),'white');draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(selected[start:start+8]):
            y=k*148;a=array(r);c=candidates.get(r['id'],pending);ann=c['annotation'] or {}
            sheet.paste(Image.fromarray(np.uint8(np.rint(composite(a,220)))).resize((128,128),Image.Resampling.NEAREST),(0,y+16))
            draw.text((0,y),str(start+k)+' '+r['id'],fill='black')
            texts=[('NAME: '+r['source_filename']+'\nOLD: '+r['old_caption'],140),
                   ('PRIOR VISUAL: '+r['previous_descriptive'],500),
                   ('NEW: '+ann.get('short_caption',c['status'])+'\n'+ann.get('detailed_caption','')+'\nSTATUS: '+c['status'],890)]
            for text,x in texts:
                lines=[]
                for line in text.splitlines():lines+=textwrap.wrap(line,54)
                draw.multiline_text((x,y+8),'\n'.join(lines),fill='black',spacing=3)
        sheet.save(pages/f'page_{start//8:03d}.png')
    atomic_json(OUT/'candidate_summary.json',{'cases':len(rows),'annotated':sum(r['id'] in candidates for r in rows),'statuses':statuses,
        'approved_for_training':0,'note':'All model candidates require visual review. Self confidence and schema validity are not accuracy.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('annotate','review'));parser.add_argument('--device',default='cuda:1');parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--memory-fraction',type=float)
    args=parser.parse_args();OUT=args.out;torch.set_num_threads(4)
    if args.memory_fraction is not None:
        if not 0<args.memory_fraction<=1:parser.error('memory fraction must be in (0, 1]')
        free,total=torch.cuda.mem_get_info(args.device)
        if free<23*1024**3:raise RuntimeError('Insufficient free GPU memory for shared annotation')
        torch.cuda.set_per_process_memory_fraction(min(args.memory_fraction,(free-3*1024**3)/total),args.device)
    if args.stage=='annotate':annotate(args.device,args.batch_size,args.limit)
    else:review()
