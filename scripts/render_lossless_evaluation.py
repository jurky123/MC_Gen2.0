"""Verify native16 evaluation outputs, summarize controls, and build a gallery."""
import argparse
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from mcgen.catalog import atomic_json, file_hash
from train_visual_reference import thumbnail
from validate_resolution import image_metrics
from evaluate_lossless_native16 import VARIANTS, SEEDS


def pixels(path):
    with Image.open(path) as im:
        x=np.array(im.convert('RGBA').resize((32,32),Image.Resampling.NEAREST)).astype(np.float32)/255
    return np.concatenate((x[...,:3]*x[...,3:],x[...,3:]),-1)


def difference(a,b):
    delta=np.abs(a-b)
    return {'premultiplied_mae':float(delta.mean()),
            'changed_pixel_fraction_at_8_of_255':float((delta.max(-1)>8/255).mean())}


def render(out,publish):
    protocol=json.loads((out/'protocol.json').read_text())
    if file_hash(out/'features.pt')!=protocol['features_sha256']:
        raise ValueError('Evaluation features changed')
    if file_hash(Path(protocol['source_run'])/'pairs.json')!=protocol['source_pairs_sha256']:
        raise ValueError('Source dataset changed during evaluation')
    rows=json.loads((out/'samples.json').read_text())
    diverse=json.loads((out/'diversity.json').read_text())
    features=torch.load(out/'features.pt',weights_only=True)
    for v in VARIANTS:
        if json.loads((out/f'progress_{v}.json').read_text())['phase']!='complete':
            raise ValueError('Generation not complete: '+v)
        metadata=json.loads((out/f'model_{v}.json').read_text())
        if file_hash(metadata['checkpoint'])!=metadata['checkpoint_sha256']:
            raise ValueError('Source checkpoint changed during evaluation')
    expected={}
    per_output=[]
    effects=[]
    for row in rows:
        with Image.open(out/'inputs'/f'{row["case_id"]}_target.png') as im:
            target=np.array(im.convert('RGBA')).repeat(2,0).repeat(2,1)
        for v in VARIANTS:
            for si in range(len(SEEDS)):
                images={}
                for mode in ('text','reference','joint'):
                    name=f'{row["case_id"]}_{v}_{mode}_s{si}.png'
                    expected[name]=32 if v=='prior32' else 16
                    path=out/'native'/name
                    images[mode]=pixels(path)
                    per_output.append({'case_id':row['case_id'],'split':row['split'],
                        'quality':row['quality_tier'],'variant':v,'mode':mode,'seed_index':si,
                        **image_metrics(path,target)})
                effects.append({'case_id':row['case_id'],'split':row['split'],'variant':v,
                                'seed_index':si,**difference(images['joint'],images['text'])})
            if row['quality_tier']=='highest_quality' and v=='after16':
                for si in range(len(SEEDS)):
                    for mode in ('text','joint'):
                        expected[f'{row["case_id"]}_{v}_{mode}_detail_s{si}.png']=16
    for row in diverse:
        for v in VARIANTS:
            for si in range(len(SEEDS)):
                expected[f'{row["case_id"]}_{v}_joint_s{si}.png']=32 if v=='prior32' else 16
    sensitivity=[]
    for prompt in ('helmet','sword','axe','shield'):
        group=[r for r in diverse if r['fixed_prompt']==prompt]
        for v in VARIANTS:
            for si in range(len(SEEDS)):
                name=f'fixedtext_{prompt}_{v}_s{si}.png'
                expected[name]=32 if v=='prior32' else 16
                plain=pixels(out/'native'/name)
                images=[pixels(out/'native'/f'{r["case_id"]}_{v}_joint_s{si}.png') for r in group]
                distances=[difference(a,b) for a,b in itertools.combinations(images,2)]
                to_text=[difference(image,plain) for image in images]
                sensitivity.append({'prompt':prompt,'variant':v,'seed_index':si,
                    'cross_reference_mae':float(np.mean([d['premultiplied_mae'] for d in distances])),
                    'cross_reference_changed_pixels':float(np.mean([d['changed_pixel_fraction_at_8_of_255'] for d in distances])),
                    'reference_vs_text_mae':float(np.mean([d['premultiplied_mae'] for d in to_text]))})
    for v in VARIANTS:
        for si in range(len(SEEDS)):
            for j in range(len(features['common_prompts'])):
                expected[f'common_{j:02d}_{v}_s{si}.png']=32 if v=='prior32' else 16
    found={p.name for p in (out/'native').glob('*.png')}
    if found!=set(expected):
        raise ValueError('Output set differs: '+str({'missing':sorted(set(expected)-found),'extra':sorted(found-set(expected))}))
    manifest=[]
    for name,size in sorted(expected.items()):
        path=out/'native'/name
        with Image.open(path) as im:
            if im.mode!='RGBA' or im.size!=(size,size):
                raise ValueError('Invalid native image: '+name)
            im.load()
        manifest.append({'file':name,'native_size':size,'sha256':file_hash(path)})
    summaries=[]
    for split in ('train','val','test'):
        for v in VARIANTS:
            for mode in ('text','reference','joint'):
                group=[m for m in per_output if (m['split'],m['variant'],m['mode'])==(split,v,mode)]
                summaries.append({'split':split,'variant':v,'mode':mode,'outputs':len(group),
                    **{key:float(np.mean([m[key] for m in group])) for key in
                       ('premultiplied_mae','alpha_iou','largest_component_fraction','near_empty')}})
    effect_summary=[]
    for split in ('train','val','test'):
        for v in VARIANTS:
            group=[m for m in effects if (m['split'],m['variant'])==(split,v)]
            effect_summary.append({'split':split,'variant':v,
                **{k:float(np.mean([m[k] for m in group])) for k in ('premultiplied_mae','changed_pixel_fraction_at_8_of_255')}})
    flow=[]
    for v in VARIANTS:
        for item in json.loads((out/f'flow_{v}.json').read_text()):
            for split in ('train','val','test'):
                ids=[i for i,r in enumerate(rows) if r['split']==split]
                scores={label:float(np.mean([values[i] for i in ids])) for label,values in item['scores'].items()}
                flow.append({'variant':v,'mode':item['mode'],'level':item['level'],'split':split,**scores,
                    'wrong_minus_matched':scores['wrong_reference_same_text']-scores['matched']})
    atomic_json(out/'metrics.json',{'summary':summaries,'per_output':per_output,
        'caveat':'Target alignment and contour proxies; do not measure MC style or decoration fidelity. All metrics compared at32 using exact2x upscaling of native16 images.'})
    atomic_json(out/'reference_sensitivity.json',{'joint_vs_text_summary':effect_summary,
        'joint_vs_text_per_output':effects,'fixed_text_reference_diversity':sensitivity,
        'flow_reference_controls':flow,
        'caveat':'Visible response to reference changes is not by itself correct design transfer. Wrong-reference control preserves text and noisy input.'})
    atomic_json(out/'generation_manifest.json',manifest)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    panels=[]
    def sheet(name,title,labels,items):
        canvas=Image.new('RGB',(14+154*len(labels),66+164*len(items)),'#111827')
        draw=ImageDraw.Draw(canvas)
        draw.text((14,10),title,font=font,fill='white')
        for c,label in enumerate(labels):
            draw.text((14+c*154,34),label,font=font,fill='#cbd5e1')
        for j,(paths,line1,line2) in enumerate(items):
            y=62+j*164
            for c,path in enumerate(paths):
                with Image.open(path) as im:
                    canvas.paste(thumbnail(im,128,native=c!=0 or len(labels)==3),(14+c*154,y))
            draw.text((14,y+132),line1[:105],font=font,fill='white')
            draw.text((14,y+147),line2[:105],font=font,fill='#aebbd0')
        canvas.save(out/'panels'/name)
        panels.append(name)
    def comparison(row):
        ident=row['case_id']
        return ([out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png']+
            [out/'native'/f'{ident}_{v}_joint_s0.png' for v in VARIANTS],
            f'{ident} | {row["captions"][0]}',row['quality_tier']+' | same short text and matched noise')
    def modes(row):
        ident=row['case_id']
        return ([out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png']+
            [out/'native'/f'{ident}_after16_{mode}_s0.png' for mode in ('text','reference','joint')]+
            [out/'native'/f'{ident}_after16_joint_s1.png'],f'{ident} | {row["captions"][0]}',row['quality_tier'])
    for split in ('train','val','test'):
        group=[r for r in rows if r['split']==split]
        for page,start in enumerate(range(0,len(group),8),1):
            part=group[start:start+8]
            sheet(f'{split}_compare_{page:02d}.png',split.upper()+' | joint comparison |32-step Heun, seed A',
                  ['HD reference','MC16 target','Original16','Previous32','Retrained16'],[comparison(r) for r in part])
            sheet(f'{split}_modes_{page:02d}.png',split.upper()+' | retrained native16 | same short text',
                  ['HD reference','MC16 target','Text only','Reference only','Text + ref A','Text + ref B'],[modes(r) for r in part])
    overview=[r for split in ('train','val','test') for r in [x for x in rows if x['split']==split][:2]]
    sheet('overview.png','Fixed overview: first two train/validation/test targets, no output filtering',
          ['HD reference','MC16 target','Original16','Previous32','Retrained16'],[comparison(r) for r in overview])
    for prompt in ('helmet','sword','axe','shield'):
        items=[]
        for row in [r for r in diverse if r['fixed_prompt']==prompt]:
            ident=row['case_id']
            paths=[out/'inputs'/f'{ident}_reference.png',out/'inputs'/f'{ident}_target.png']
            paths+=[out/'native'/f'{ident}_{v}_joint_s0.png' for v in VARIANTS]
            paths+=[out/'native'/f'fixedtext_{prompt}_after16_s0.png']
            items.append((paths,row['captions'][0],f'{row["split"]} | identical prompt: {prompt}; same Gaussian noise'))
        sheet(f'fixed_{prompt}.png',f'Fixed text: {prompt} | four different reference designs | seed A',
              ['HD reference','MC16 target','Original16 joint','Previous32 joint','Retrained16 joint','Retrained16 text'],items)
    for si in range(len(SEEDS)):
        items=[([out/'native'/f'common_{j:02d}_{v}_s{si}.png' for v in VARIANTS],prompt,'same seed and common text embedding')
               for j,prompt in enumerate(features['common_prompts'])]
        sheet(f'common_s{si}.png',f'Common text prompts | seed {si+1}',['Original16','Previous32','Retrained16'],items)
    build_html(out,rows,diverse,features['common_prompts'],panels)
    body=(out/'index.html').read_text()
    js=re.search(r'<script>(.*?)</script>',body,re.S).group(1)
    check=out/'gallery_check.js'
    check.write_text(js)
    subprocess.run(['node','--check',str(check)],check=True,capture_output=True,text=True)
    check.unlink()
    atomic_json(out/'verification.json',{'complete':True,'primary_cases':len(rows),
        'unique_primary_targets':len({r['target_hash'] for r in rows}),
        'primary_outputs':len(per_output),'native_pngs_verified':len(manifest),'panels':panels,
        'all_native_dimensions_and_rgba_valid':True,'javascript_syntax':'passed','new_training':False})
    if publish:
        publish.mkdir(parents=True,exist_ok=True)
        for directory in ('native','inputs','panels'):
            shutil.copytree(out/directory,publish/directory,dirs_exist_ok=True)
        shutil.copy2(out/'index.html',publish/'index.html')
    print(json.dumps({'native_outputs':len(manifest),'panels':len(panels),'summary':summaries,
                      'reference_sensitivity':effect_summary},indent=2),flush=True)


def build_html(out,rows,diverse,common,panels):
    payload={'rows':[{'id':r['case_id'],'split':r['split'],'name':r['captions'][0],
                      'detail':r['captions'][-1],'hq':r['quality_tier']=='highest_quality'} for r in rows],
             'diverse':[{'id':r['case_id'],'split':r['split'],'name':r['captions'][0],
                         'prompt':r['fixed_prompt']} for r in diverse], 'common':common,'panels':panels}
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>16×16重训测评</title>
<style>body{background:#111827;color:#e5e7eb;font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}a{color:#93c5fd}.controls{position:sticky;top:0;background:#1f2937;padding:12px;z-index:2;border-radius:8px}select{padding:8px;margin:4px 12px 4px 4px}article{background:#1f2937;margin:18px 0;padding:16px;border-radius:10px}h3{font-size:15px}.grid{display:grid;gap:12px;overflow-x:auto}.grid img{width:128px;height:128px;object-fit:contain;image-rendering:pixelated;background:#edf0f4}.grid img.hd{image-rendering:auto}details img{max-width:100%}.note{color:#b6c4d8}</style>
<p><a href="/">返回数据压缩对照</a></p><h1>无损16×16数据重训 · 生成测评</h1>
<p>24个训练目标、24个验证目标、16个测试目标；两个种子。全部目标均可无损恢复16格，未加入32→16有损数据。</p>
<p class="note">本轮两个最佳权重与最后权重的模型参数完全相同，均为270步。“上一轮32格”用于结果参照，数据构成不同，不能据此单独判断分辨率的因果影响。训练外指本轮微调未使用。</p>
<div class="controls"><label>内容<select id="view"><option value="compare">三版模型对照</option><option value="latest">本轮三种任务</option><option value="fixed">固定文字、更换参考图</option><option value="common">常见文生图</option></select></label>
<label>划分<select id="split"><option value="test">测试集</option><option value="val">验证集</option><option value="train">训练集</option><option value="all">全部</option></select></label>
<label>任务<select id="mode"><option value="joint">图文联合</option><option value="text">纯文本</option><option value="reference">纯参考图</option></select></label>
<label>种子<select id="seed"><option value="0">A</option><option value="1">B</option></select></label><label><input type="checkbox" id="detail">本轮任务页使用精标描述</label></div>
<div id="cards"></div><h2>可直接打开的静态对照图</h2><div id="panels"></div><script>
const data=PAYLOAD,variants=["before16","prior32","after16"],labels=["原16×16","上一轮32×32","本轮重训16×16"];
function card(title,images){const article=document.createElement("article"),h=document.createElement("h3"),grid=document.createElement("div");h.textContent=title;article.append(h);grid.className="grid";grid.style.gridTemplateColumns="repeat("+images.length+",minmax(128px,1fr))";for(const [label,path,hd] of images){const cell=document.createElement("div"),p=document.createElement("p"),img=document.createElement("img");p.textContent=label;img.src=path;img.alt=label;img.loading="lazy";if(hd)img.className="hd";cell.append(p,img);grid.append(cell)}article.append(grid);document.getElementById("cards").append(article)}
function draw(){document.getElementById("cards").replaceChildren();const view=document.getElementById("view").value,split=document.getElementById("split").value,mode=document.getElementById("mode").value,seed=document.getElementById("seed").value,detail=document.getElementById("detail").checked;
if(view==="common"){data.common.forEach((prompt,j)=>card(prompt,variants.map((v,k)=>[labels[k],"native/common_"+String(j).padStart(2,"0")+"_"+v+"_s"+seed+".png"])));return}
if(view==="fixed"){for(const row of data.diverse){if(split!=="all"&&row.split!==split)continue;card("固定文字："+row.prompt+" | "+row.name+" | "+row.split,[["参考图","inputs/"+row.id+"_reference.png",true],["MC目标","inputs/"+row.id+"_target.png"],...variants.map((v,k)=>[labels[k]+"图文","native/"+row.id+"_"+v+"_joint_s"+seed+".png"]),["本轮纯文字","native/fixedtext_"+row.prompt+"_after16_s"+seed+".png"]])}return}
for(const row of data.rows){if(split!=="all"&&row.split!==split)continue;const images=[["参考图","inputs/"+row.id+"_reference.png",true],["MC目标16×16","inputs/"+row.id+"_target.png"]];if(view==="compare"){variants.forEach((v,k)=>images.push([labels[k],"native/"+row.id+"_"+v+"_"+mode+"_s"+seed+".png"]))}else{for(const [task,label] of [["text","纯文本"],["reference","纯参考图"],["joint","图文联合"]]){const suffix=detail&&row.hq&&task!=="reference"?"_detail":"";images.push([label+(suffix?" · 精标文本":""),"native/"+row.id+"_after16_"+task+suffix+"_s"+seed+".png"])}}card(row.id+" | "+row.name+(view==="latest"&&detail&&row.hq?" | "+row.detail:""),images)}}
for(const id of ["view","split","mode","seed","detail"])document.getElementById(id).addEventListener("change",draw);for(const name of data.panels){const d=document.createElement("details"),s=document.createElement("summary"),a=document.createElement("a"),img=document.createElement("img");s.textContent=name;a.href="panels/"+name;img.src=a.href;img.loading="lazy";a.append(img);d.append(s,a);document.getElementById("panels").append(d)}draw();</script></html>'''
    (out/'index.html').write_text(document.replace('PAYLOAD',data))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/native16_lossless_eval_20261107'))
    parser.add_argument('--publish',type=Path,default=Path('outputs/native16_data_review_20261107/evaluation'))
    args=parser.parse_args()
    render(args.out,args.publish)
