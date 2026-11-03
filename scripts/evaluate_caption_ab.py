"""Same prompts, seeds and targets for both caption-supervision arms."""
import argparse
import gc
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from mcgen.catalog import read_catalog,file_hash,atomic_json
from mcgen.runtime import Runtime,load_checkpoint
from mcgen.semantic_captions import filename_words
from mcgen.evaluation import metrics,summarize
from mcgen.images import composite
from audit_item_captions import array


def evaluate(root,catalogs,runs,out,device,candidate='new'):
    if out.exists():raise ValueError('Keep evaluation outputs immutable; choose a new directory')
    old,old_meta=read_catalog(catalogs/'old');new,new_meta=read_catalog(catalogs/candidate)
    assert {(r['target_hash'],r['split'],r['lineage']) for r in old}=={(r['target_hash'],r['split'],r['lineage']) for r in new}
    if candidate=='new':
        assert [(r['id'],r['target_hash'],r['split']) for r in old]==[(r['id'],r['target_hash'],r['split']) for r in new]
    assert [r['prompt'] for r in old if r['split']=='val']==[r['prompt'] for r in new if r['split']=='val']
    configs=[];init=[];steps=[]
    for arm in ['old',candidate]:
        state=load_checkpoint(runs/arm/'latest.pt');configs.append(state['config']);init.append(state['initialization']['sha256']);steps.append(state['step']);del state
    assert configs[0]==configs[1] and init[0]==init[1] and steps==[600,600]
    val=[r for r in old if r['split']=='val'];rng=random.Random(20261107);rng.shuffle(val)
    labels={r['id']:r for r in map(json.loads,(root/'refined/semantic/candidates.jsonl').read_text().splitlines())}
    reviewed=json.loads((root/'quality_review.json').read_text())['ratings']
    rich_ids={r['id'] for r in reviewed if r['split']=='val' and r['new_rating']=='plausible'}
    chosen=[]
    for kind,subset in [('name',val[:64]),('rich',[r for r in val if r['id'] in rich_ids])]:
        for r in subset:
            prompt=r['prompt'] if kind=='name' else labels[r['id']]['combined_text']
            chosen.append({**r,'prompt_kind':kind,'evaluation_prompt':prompt,'seed':20261107+int(r['target_hash'][:6],16)})
    out.mkdir(parents=True);(out/'images').mkdir()
    atomic_json(out/'protocol.json',{'cases':[{'id':r['id'],'target_hash':r['target_hash'],'kind':r['prompt_kind'],'prompt':r['evaluation_prompt'],'seed':r['seed']} for r in chosen],
        'arms':['old',candidate],'steps':32,'solver':'heun','text_scale':2,'endpoint_training_step':600,'initialization_sha256':init[0],
        'name_selection':'64 seeded validation targets, fixed common filename prompts',
        'rich_selection':'Frozen-review validation cases whose rich captions were rated plausible before training',
        'warning':'Native-grid error metrics do not measure semantic accuracy or the full Minecraft image distribution.'})
    all_rows=[];generated={}
    for arm,meta in [('old',old_meta),(candidate,new_meta)]:
        while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
        free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
        runtime=Runtime(runs/arm/'latest.pt',device,device)
        assert runtime.metadata['catalog_fingerprint']==meta['fingerprint']
        images=[]
        for start in range(0,len(chosen),8):
            batch=chosen[start:start+8]
            images.extend(runtime.generate([r['evaluation_prompt'] for r in batch],seeds=[r['seed'] for r in batch],steps=32,solver='heun',text_scale=2))
            print(json.dumps({'arm':arm,'generated':len(images),'total':len(chosen)}),flush=True)
        for i,(r,img) in enumerate(zip(chosen,images)):
            Image.fromarray(img).save(out/'images'/f'{i:03d}_{arm}.png')
            score=metrics(img,array(r));score['seam_mae']=None
            all_rows.append({'id':r['id'],'target_hash':r['target_hash'],'lineage':r['lineage'],'mode':arm+':'+r['prompt_kind'],'metrics':score})
        generated[arm]=images;del runtime;gc.collect();torch.cuda.empty_cache()
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    atomic_json(out/'summary.json',summarize(all_rows))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for start in range(0,len(chosen),8):
        sheet=Image.new('RGB',(1100,8*160+40),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,6),f'Prompt | Target | Old-caption training | {candidate} caption training',font=font,fill='black')
        for k,r in enumerate(chosen[start:start+8]):
            i=start+k;y=40+k*160
            prompt=r['prompt_kind']+': '+r['evaluation_prompt'];line='';lines=[]
            for char in prompt:
                if font.getlength(line+char)>480:lines.append(line);line=char
                else:line+=char
            lines.append(line);draw.multiline_text((8,y),'\n'.join(lines),font=font,fill='black',spacing=3)
            for x,img in [(520,array(r)),(700,generated['old'][i]),(880,generated[candidate][i])]:
                sheet.paste(Image.fromarray(np.uint8(np.rint(composite(img,128)))).resize((128,128),Image.Resampling.NEAREST),(x,y))
            draw.line((0,y+154,1100,y+154),fill='#ddd')
        sheet.save(out/f'comparison_{start//8:02d}.png')
    print(json.dumps({'evaluation_complete':True,'cases_per_arm':len(chosen)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--catalogs',required=True,type=Path)
    p.add_argument('--runs',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cuda:1')
    p.add_argument('--candidate',choices=['new','mixed'],default='new')
    a=p.parse_args();torch.set_num_threads(4);evaluate(a.root,a.catalogs,a.runs,a.out,a.device,a.candidate)
