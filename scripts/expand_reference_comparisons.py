"""More seen/unseen targets, two seeds, and paired original-HD reference variants."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.images import decode
from mcgen.reference_adapter import backbone_digest
from mcgen.runtime import Runtime,load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from evaluate_reference_adapter import prepare,RUN


def prepare_expansion(out,count,paired,seed):
    if count<1 or not 0<=paired<=count:raise ValueError('require count>0 and 0<=paired<=count')
    excluded=[Path('outputs/reference_adapter_eval_20261104/samples.json'),
              Path('outputs/zero_shot_reference_20261104/cases.json')]
    prepare(out,seed,per_split=count,exclude_manifests=excluded)
    rows=json.load(open(out/'samples.json'));alternates=[]
    for split in ('train','test'):
        if paired==0:continue
        candidates=[r for r in rows if r['split']==split and r['alternate_references']]
        if len(candidates)<paired:raise ValueError('not enough paired reference variants')
        # Spread paired comparisons across label groups before filling.
        chosen=[];counts=Counter()
        for r in candidates:
            if counts[r['category']]:continue
            chosen.append(r);counts[r['category']]+=1
            if len(chosen)==paired:break
        for r in candidates:
            if len(chosen)==paired:break
            if r not in chosen:chosen.append(r)
        for r in chosen:
            alt=r['alternate_references'][0];ident=r['case_id']+'_alt'
            if file_hash(alt['reference_file'])!=alt['reference_sha256']:raise ValueError('changed alternate reference')
            with Image.open(alt['reference_file']) as im:im.convert('RGBA').save(out/'inputs'/f'{ident}_reference.png')
            alternates.append({**r,**alt,'case_id':ident,'base_case_id':r['case_id']})
    atomic_json(out/'alternate_samples.json',alternates)
    protocol=json.load(open(out/'protocol.json'))
    protocol.update(comparison_checkpoints=['best_flow.pt'],generation_seeds=[202611042,202611043],
                    primary_modes=['text','image','joint'],paired_reference_targets={'train':paired,'test':paired},
                    expected_native_images=2*(3*len(rows)+2*len(alternates)),
                    selection='Fresh filename-stratified targets; prior reviewed targets excluded; no generated-output filtering.')
    atomic_json(out/'protocol.json',protocol)


def generate(out,device,batch_size):
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.65,device)
    started=time.time();rows=json.load(open(out/'samples.json'));alternates=json.load(open(out/'alternate_samples.json'))
    runtime=Runtime(RUN/'training/best_flow.pt',device=device)
    primary_images=[Image.open(out/'inputs'/f"{r['case_id']}_reference.png").copy() for r in rows]
    all_rows=rows+alternates
    all_images=primary_images+[Image.open(out/'inputs'/f"{r['case_id']}_reference.png").copy() for r in alternates]
    prompts=conditioned_prompts([r['prompt'] for r in rows],runtime.config)
    text=[]
    for start in range(0,len(prompts),16):
        h,_=runtime.encoder.encode(prompts[start:start+16],'cpu');text.append(h)
    features={'text':torch.cat(text)}
    for mode in ('image','joint'):
        chunks=[]
        for start in range(0,len(all_rows),16):
            group=all_rows[start:start+16]
            captions=['']*len(group) if mode=='image' else conditioned_prompts([r['prompt'] for r in group],runtime.config)
            h,_=runtime.encoder.encode_multimodal(captions,all_images[start:start+16],'cpu');chunks.append(h)
            atomic_json(out/'progress.json',{'phase':'encoding','mode':mode,'done':start+len(group),'total':len(all_rows),'seconds':time.time()-started})
        features[mode]=torch.cat(chunks)
        print('ENCODED',mode,len(all_rows),round(time.time()-started,1),flush=True)
    identity=runtime.encoder.embedding.multimodal_identity
    torch.save(features,out/'embeddings.pt')
    del runtime.encoder;gc.collect();torch.cuda.empty_cache()
    original_state=load_checkpoint('checkpoints/mcgen-step4128.pt');base_cfg=Experiment.from_dict(original_state['config'])
    original=PixelFlow(base_cfg.model).to(device);original.load_state_dict(original_state['model']);original.eval()
    assert backbone_digest(original)==backbone_digest(runtime.model)
    total_images=0;text_exact=True
    for seed_index in range(2):
        for mode in ('text','image','joint'):
            selected=rows if mode=='text' else all_rows
            for start in range(0,len(selected),batch_size):
                batch=selected[start:start+batch_size];h=features[mode][start:start+len(batch)].to(device)
                mask=torch.ones(h.shape[:2],dtype=torch.bool,device=device);seeds=[r['generation_seed']+seed_index for r in batch]
                model=original if mode=='text' else runtime.model
                values=sample(model,h,mask,seeds=seeds,steps=32,text_scale=2.,embedding_adapter=mode!='text')
                if mode=='text':
                    retained=sample(runtime.model,h,mask,seeds=seeds,steps=32,text_scale=2.)
                    text_exact=text_exact and torch.equal(values,retained)
                    if not text_exact:raise RuntimeError('original text output changed')
                for row,value in zip(batch,values):
                    Image.fromarray(decode(value)).save(out/'native'/f"{row['case_id']}_{mode}_s{seed_index}.png")
                total_images+=len(batch)
                atomic_json(out/'progress.json',{'phase':'generation','mode':mode,'seed_index':seed_index,
                             'done_in_mode':start+len(batch),'total_in_mode':len(selected),'images_written':total_images,'seconds':time.time()-started})
            print('GENERATED',mode,'seed',seed_index,round(time.time()-started,1),flush=True)
    expected=json.load(open(out/'protocol.json'))['expected_native_images'];assert total_images==expected
    atomic_json(out/'results.json',{'checkpoint':str(RUN/'training/best_flow.pt'),'checkpoint_step':runtime.metadata['step'],
                'checkpoint_sha256':file_hash(RUN/'training/best_flow.pt'),'multimodal_identity':identity,
                'targets':len(rows),'alternate_references':len(alternates),'native_images':total_images,
                'text_generation_exact':text_exact,'backbone_exact':True,'seeds':[202611042,202611043],
                'counts':dict(Counter(r['split'] for r in rows)),'seconds':time.time()-started,
                'limitations':'Qualitative filename-stratified sampling; no overall quality or accuracy estimate.'})
    atomic_json(out/'progress.json',{'phase':'complete','images_written':total_images,'seconds':time.time()-started})
    print('COMPLETE',total_images,round(time.time()-started,1),flush=True)


def tile(path,size,photograph=False):
    with Image.open(path) as im:
        im=im.convert('RGBA');bg=Image.new('RGBA',im.size,(239,241,245,255));bg.alpha_composite(im)
        return bg.convert('RGB').resize((size,size),Image.Resampling.LANCZOS if photograph else Image.Resampling.NEAREST)


def render(out):
    rows=json.load(open(out/'samples.json'));alternates=json.load(open(out/'alternate_samples.json'));results=json.load(open(out/'results.json'))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    labels=['HD reference','MC target','Text only','Image only','Image + text','Image seed 2','Joint seed 2']
    panels=[]
    for split in ('train','test'):
        selected=[r for r in rows if r['split']==split];pages=(len(selected)+7)//8
        for page in range(pages):
            canvas=Image.new('RGB',(1008,1360),'#111827');draw=ImageDraw.Draw(canvas)
            draw.text((18,14),f'MCGen | {split.upper()} | new samples {page*8+1}-{min((page+1)*8,len(selected))}/{len(selected)} | adapter step {results["checkpoint_step"]}',font=font,fill='white')
            for col,label in enumerate(labels):draw.text((18+140*col,43),label,font=font,fill='#e5e7eb')
            for line,r in enumerate(selected[page*8:page*8+8]):
                ident=r['case_id'];y=72+line*160
                paths=[out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png']+[out/'native'/f'{ident}_{mode}_s{s}.png' for mode,s in [('text',0),('image',0),('joint',0),('image',1),('joint',1)]]
                for col,path in enumerate(paths):canvas.paste(tile(path,128,col==0),(18+col*140,y))
                draw.text((18,y+131),f'{ident} | {r["prompt"]} | {r["reference_generator"]}',font=small,fill='#e5e7eb')
            name=f'{split}_comparison_{page+1:02d}.png';canvas.save(out/name);panels.append(name)
        alt_selected=[r for r in alternates if r['split']==split]
        for page in range((len(alt_selected)+7)//8):
            canvas=Image.new('RGB',(1148,1360),'#111827');draw=ImageDraw.Draw(canvas)
            draw.text((18,14),f'MCGen | {split.upper()} | same target, alternate HD reference | page {page+1}',font=font,fill='white')
            for col,label in enumerate(['HD ref A','HD ref B','MC target','Text only','Image A','Image B','Joint A','Joint B']):
                draw.text((18+col*140,43),label,font=font,fill='#e5e7eb')
            for line,r in enumerate(alt_selected[page*8:page*8+8]):
                a=r['base_case_id'];b=r['case_id'];y=72+line*160
                paths=[out/'inputs'/f'{a}_reference.png',out/'inputs'/f'{b}_reference.png',out/'inputs'/f'{a}_target.png',out/'native'/f'{a}_text_s0.png']+[out/'native'/f'{i}_{mode}_s0.png' for mode in ('image','joint') for i in (a,b)]
                for col,path in enumerate(paths):canvas.paste(tile(path,128,col<2),(18+col*140,y))
                draw.text((18,y+131),f'{a} | {r["prompt"]}',font=small,fill='#e5e7eb')
            name=f'{split}_references_{page+1:02d}.png';canvas.save(out/name);panels.append(name)
    # Compact four-column overviews, 16 targets per page, for browsing in chat.
    for split in ('train','test'):
        selected=[r for r in rows if r['split']==split]
        for page in range((len(selected)+15)//16):
            canvas=Image.new('RGB',(1080,1260),'#111827');draw=ImageDraw.Draw(canvas)
            draw.text((18,14),f'MCGen | {split.upper()} | fresh targets | overview {page+1}',font=font,fill='white')
            for block in range(2):
                for col,label in enumerate(['HD reference','MC target','Image only','Image + text']):draw.text((18+block*536+col*126,43),label,font=font,fill='#e5e7eb')
            for i,r in enumerate(selected[page*16:page*16+16]):
                block=i//8;line=i%8;y=70+line*146;ident=r['case_id']
                paths=[out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png',out/'native'/f'{ident}_image_s0.png',out/'native'/f'{ident}_joint_s0.png']
                for col,path in enumerate(paths):canvas.paste(tile(path,112,col==0),(18+block*536+col*126,y))
                draw.text((18+block*536,y+115),f'{ident}: {r["prompt"]}',font=small,fill='#e5e7eb')
            name=f'{split}_overview_{page+1:02d}.png';canvas.save(out/name);panels.append(name)
    atomic_json(out/'panels.json',panels)
    payload=json.dumps({'rows':rows,'alternates':{r['base_case_id']:r for r in alternates}},ensure_ascii=False).replace('</','<\/')
    gallery='''<!doctype html><meta charset="utf-8"><title>MCGen — more seen/unseen comparisons</title>
<style>body{background:#111827;color:#e5e7eb;font:15px system-ui;margin:24px}h1{font-size:24px}select{padding:8px;margin:4px;background:#263247;color:white;border:1px solid #526078}article{border:1px solid #364152;padding:14px;margin:14px 0;max-width:940px}h2{font-size:16px;margin:0 0 12px}.images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}figure img{width:144px;height:144px;object-fit:contain;background:#eff1f5;image-rendering:pixelated}figure:first-child img{image-rendering:auto}figcaption{margin-top:5px;color:#aab5c6}a{color:#93c5fd}</style>
<h1>训练内 / 训练未见：扩大对照</h1><p>128 个新物品；每个两种种子；32 个物品额外比较另一版高清参考图。原始 16×16 RGBA，无输出筛选。</p>
<label>分组 <select id="split"><option value="test">训练未见</option><option value="train">训练内</option><option value="all">全部</option></select></label>
<label>文件名分组 <select id="category"><option value="all">全部</option></select></label>
<label>种子 <select id="seed"><option value="0">种子 1</option><option value="1">种子 2</option></select></label>
<label>参考图 <select id="reference"><option value="primary">原版</option><option value="alternate">另一版（仅显示有对照的物品）</option></select></label>
<p id="count"></p><main id="cards"></main><details><summary>完整对照面板</summary><div id="panels"></div></details>
<script>const data=PAYLOAD;const $=id=>document.getElementById(id);const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
for(const c of [...new Set(data.rows.map(r=>r.category))].sort())$('category').add(new Option(c,c));
function render(){const split=$('split').value,category=$('category').value,seed=$('seed').value,alt=$('reference').value==='alternate';const rows=data.rows.filter(r=>(split==='all'||r.split===split)&&(category==='all'||r.category===category)&&(!alt||data.alternates[r.case_id]));$('count').textContent=`显示 ${rows.length} 个物品`;$('cards').innerHTML=rows.map(r=>{const ref=alt?data.alternates[r.case_id]:r;const paths=[['参考图',`inputs/${ref.case_id}_reference.png`],['MC 原图',`inputs/${r.case_id}_target.png`],['纯文本',`native/${r.case_id}_text_s${seed}.png`],['纯参考图',`native/${ref.case_id}_image_s${seed}.png`],['图文联合',`native/${ref.case_id}_joint_s${seed}.png`]];return `<article><h2>${esc(r.case_id+' · '+r.prompt+' · '+ref.reference_generator)}</h2><div class="images">${paths.map(([name,path])=>`<figure><a href="${path}"><img loading="lazy" src="${path}"></a><figcaption>${name}</figcaption></figure>`).join('')}</div></article>`}).join('')}
for(const id of ['split','category','seed','reference'])$(id).onchange=render;render();const panels=PANELS;$('panels').innerHTML=panels.map(p=>`<p><a href="${p}">${esc(p)}</a></p>`).join('');</script>'''
    gallery=gallery.replace('128 个新物品',f'{len(rows)} 个新物品').replace('32 个物品额外',f'{len(alternates)} 个物品额外')
    gallery=gallery.replace('PAYLOAD',payload).replace('PANELS',json.dumps(panels));(out/'index.html').write_text(gallery)
    print('GALLERY READY',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['prepare','generate','render','all'],default='all')
    p.add_argument('--per-split',type=int,default=64);p.add_argument('--paired-per-split',type=int,default=16);p.add_argument('--seed',type=int,default=202611044)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
    if a.phase in ('prepare','all'):prepare_expansion(a.out,a.per_split,a.paired_per_split,a.seed)
    if a.phase in ('generate','all'):generate(a.out,a.device,a.batch_size)
    if a.phase in ('render','all'):render(a.out)
