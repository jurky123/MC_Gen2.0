"""Review fitting with saved PNGs, including oracle and no-reference controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mcgen.catalog import atomic_json, file_hash
from train_visual_reference import thumbnail

ARMS=('qwen_hd','oracle_target')


def load_pixels(path):
    with Image.open(path) as im:
        if im.mode!='RGBA' or im.size!=(16,16):raise ValueError('Expected native16 RGBA: '+str(path))
        x=np.asarray(im).astype(np.float32)/255
    return np.concatenate((x[...,:3]*x[...,3:],x[...,3:]),-1)


def main(root,step,publish):
    rows=json.loads((root/'samples.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert all(r['split']=='train' for r in rows)
    assert file_hash(protocol['source_checkpoint'])==protocol['source_checkpoint_sha256']
    assert file_hash(root/'features.pt')==protocol['features_sha256']
    targets=np.stack([load_pixels(root/'inputs'/f'{r["case_id"]}_target.png') for r in rows])
    categories=[r['fixed_prompt'] for r in rows]
    perm=[]
    for i,c in enumerate(categories):
        ids=[j for j,v in enumerate(categories) if v==c]
        perm.append(ids[(ids.index(i)+1)%len(ids)])
    summaries=[];per_image=[];expected=set()
    for arm in ARMS:
        for metrics in sorted((root/arm).glob('eval_*/metrics.json')):
            report=json.loads(metrics.read_text())
            current=report['step']
            if current>step:continue
            for key in report['generation']:
                task,seed=key.rsplit('_s',1)
                group=[]
                for i,row in enumerate(rows):
                    name=f'{arm}_{current:06d}_{row["case_id"]}_{task}_s{seed}.png'
                    path=root/'generated'/name
                    prediction=load_pixels(path)
                    expected.add(name)
                    truth=targets[perm[i]] if task=='wrong_reference' else targets[i]
                    a,b=prediction[...,3]>=.5,truth[...,3]>=.5
                    union=a|b
                    choices=[j for j,c in enumerate(categories) if c==categories[i]]
                    retrieved=min(choices,key=lambda j:np.abs(prediction-targets[j]).mean())
                    wanted=perm[i] if task=='wrong_reference' else i
                    error=np.abs(prediction-truth)
                    value={'arm':arm,'step':current,'mode':task,'seed_index':int(seed),'case_id':row['case_id'],
                        'mae':float(error.mean()),'alpha_iou':float((a&b).sum()/union.sum()) if union.any() else 1.,
                        'foreground_mae':float(error[union].mean()) if union.any() else 0.,
                        'correct_design':retrieved==wanted,'retrieved_case_id':rows[retrieved]['case_id']}
                    per_image.append(value);group.append(value)
                summaries.append({'arm':arm,'step':current,'mode':task,'seed_index':int(seed),
                    **{k:float(np.mean([v[k] for v in group])) for k in ('mae','alpha_iou','foreground_mae')},
                    'correct_designs':sum(v['correct_design'] for v in group),'targets':len(group)})
    for arm in ARMS:
        assert (root/arm/f'eval_{step:06d}/metrics.json').exists()
        if step==protocol['train_steps']:
            assert json.loads((root/arm/'status.json').read_text())['complete']
    panels=root/'panels';panels.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    panel_names=[]
    def output(arm,current,row,mode='joint',seed=0):
        return root/'generated'/f'{arm}_{current:06d}_{row["case_id"]}_{mode}_s{seed}.png'
    def sheet(name,title,labels,items):
        image=Image.new('RGB',(14+154*len(labels),66+164*len(items)),'#111827')
        draw=ImageDraw.Draw(image);draw.text((14,10),title,font=font,fill='white')
        for j,label in enumerate(labels):draw.text((14+j*154,34),label,font=font,fill='#cbd5e1')
        for i,(paths,line1,line2) in enumerate(items):
            y=62+i*164
            for j,path in enumerate(paths):
                with Image.open(path) as im:image.paste(thumbnail(im,128,native=j>0),(14+j*154,y))
            draw.text((14,y+132),line1[:110],font=font,fill='white')
            draw.text((14,y+147),line2[:110],font=font,fill='#aebbd0')
        image.save(panels/name);panel_names.append(name)
    for category in dict.fromkeys(categories):
        selected=[r for r in rows if r['fixed_prompt']==category]
        items=[]
        for row in selected:
            ident=row['case_id']
            items.append(([root/'inputs'/f'{ident}_reference.png',root/'inputs'/f'{ident}_target.png',
                           output('qwen_hd',0,row),output('qwen_hd',step,row),output('qwen_hd',step,row,'reference'),
                           output('oracle_target',step,row)],row['captions'][0],
                          f'Train only | same prompt: {category} | step{step} | CFG1 | oracle leaks target'))
        sheet(f'{category}_step{step:06d}.png',f'{category} | four training designs | same category text and Gaussian noise',
              ['HD reference','MC16 target','Before fitting','Qwen HD joint','Qwen HD image','Oracle joint'],items)
    shared_steps=[s for s in (0,250,1000,2000) if s<=step and all((root/a/f'eval_{s:06d}/metrics.json').exists() for a in ARMS)]
    for arm in ARMS:
        items=[]
        for category in dict.fromkeys(categories):
            row=next(r for r in rows if r['fixed_prompt']==category)
            items.append(([root/'inputs'/f'{row["case_id"]}_target.png']+[output(arm,s,row) for s in shared_steps],
                          row['captions'][0],'first selected design per category; fixed seed; CFG1'))
        sheet(f'{arm}_history.png',arm+' | fitting trajectory',['MC target']+[f'Step{s}' for s in shared_steps],items)
    if step==2000:
        for page,start in enumerate(range(0,len(rows),8),1):
            items=[]
            for row in rows[start:start+8]:
                ident=row['case_id']
                items.append(([root/'inputs'/f'{ident}_reference.png',root/'inputs'/f'{ident}_target.png',
                    output('qwen_hd',step,row),output('qwen_hd',step,row,'joint_cfg2'),
                    output('oracle_target',step,row),output('oracle_target',step,row,'joint_cfg2')],row['captions'][0],
                    'same seed | CFG1 isolates conditional fitting | oracle is target leakage control'))
            sheet(f'cfg_compare_{page:02d}.png','Guidance check after fitting',
                  ['HD reference','MC16 target','Qwen CFG1','Qwen CFG2','Oracle CFG1','Oracle CFG2'],items)
    results={'step':step,'complete':step==protocol['train_steps'],'summaries':summaries,'per_image':per_image,
        'metric_space':'actual saved uint8 PNGs, straight RGBA decoded and premultiplied before errors; training evaluation JSON has pre-decode latent metrics',
        'design_retrieval':'nearest of four same-category training targets; chance25%; no-reference samples share identical category and noise',
        'scope':'16 training targets only; no evidence of generalization. Oracle deliberately leaks target pixels.',
        'panels':panel_names}
    atomic_json(root/f'results_{step:06d}.json',results)
    if step==2000:atomic_json(root/'results.json',results)
    if step==2000:
        assert {p.name for p in (root/'generated').glob('*.png')}==expected
        runs=[json.loads((root/a/'run.json').read_text()) for a in ARMS]
        assert len({r['initial_model_sha256'] for r in runs})==1
    images=[{'file':name,'sha256':file_hash(root/'generated'/name)} for name in sorted(expected)]
    atomic_json(root/f'generation_manifest_{step:06d}.json',images)
    html(root,rows,step,panel_names)
    js=re.search(r'<script>(.*?)</script>',(root/'index.html').read_text(),re.S).group(1)
    check=root/'page_check.js';check.write_text(js)
    subprocess.run(['node','--check',str(check)],check=True,capture_output=True,text=True);check.unlink()
    if publish:
        publish.mkdir(parents=True,exist_ok=True)
        for name in ('inputs','generated','panels'):
            shutil.copytree(root/name,publish/name,dirs_exist_ok=True)
        shutil.copy2(root/'index.html',publish/'index.html')
    atomic_json(root/f'verification_{step:06d}.json',{'complete':True,'fitting_complete':step==2000,
        'native_pngs_verified':len(expected),'all_targets_from_training':True,'source_checkpoint_unchanged':True,
        'javascript_syntax':'passed','oracle_is_target_leakage_control':True})
    print(json.dumps([s for s in summaries if s['step']==step],indent=2),flush=True)


def html(root,rows,step,panels):
    data=json.dumps({'rows':[{'id':r['case_id'],'category':r['fixed_prompt'],'name':r['captions'][0]} for r in rows],
                     'step':step,'panels':panels},ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>16样本拟合诊断</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:1100px;margin:24px auto;padding:0 16px}a{color:#93c5fd}.warning{background:#713f12;padding:12px;border-radius:8px}.controls{position:sticky;top:0;padding:12px;background:#1f2937;z-index:2}select{padding:8px;margin:4px 16px 4px 4px}article{margin:20px 0;padding:14px;background:#1f2937;border-radius:10px}h3{font-size:15px}.grid{display:grid;grid-template-columns:repeat(6,minmax(128px,1fr));gap:12px;overflow-x:auto}.grid img{width:128px;height:128px;object-fit:contain;image-rendering:pixelated;background:#edf0f4}.grid img.hd{image-rendering:auto}details img{max-width:100%}</style>
<p><a href="/">数据对照</a> · <a href="/evaluation/">上一轮测评</a></p><h1>16个训练目标 · 拟合诊断</h1>
<p>四种类别各四个设计，固定类别文字和同一生成噪声，不能依靠独有名称或不同种子区分目标。两个实验使用相同初始化和优化设置。</p>
<p class="warning">目标像素对照直接包含答案，只用于检查训练/注入能力，不能用来评价或部署高清参考图生成。这里仅测试训练集记忆能力。</p>
<div class="controls"><label>类别<select id="category"><option value="helmet">头盔</option><option value="sword">剑</option><option value="axe">斧</option><option value="shield">盾牌</option><option value="all">全部</option></select></label>
<label>步数<select id="step"></select></label><label>任务<select id="mode"><option value="joint">固定文字+参考图</option><option value="reference">仅参考图</option><option value="text">仅类别文字</option></select></label>
<label>种子<select id="seed"><option value="0">A</option><option value="1">B（2000步）</option></select></label><label><input type="checkbox" id="cfg">CFG2（2000步）</label></div><div id="cards"></div><h2>静态对照</h2><div id="panels"></div>
<script>const data=DATA;const steps=[0,250,1000,2000].filter(s=>s<=data.step);for(const s of steps){const o=document.createElement("option");o.value=s;o.textContent=s;document.getElementById("step").append(o)}document.getElementById("step").value=data.step;
function draw(){const parent=document.getElementById("cards");parent.replaceChildren();const category=document.getElementById("category").value,current=Number(document.getElementById("step").value),mode=document.getElementById("mode").value;const seed=current===2000?document.getElementById("seed").value:"0",cfg=current===2000&&mode!=="text"&&document.getElementById("cfg").checked;document.getElementById("seed").disabled=current!==2000;document.getElementById("cfg").disabled=current!==2000||mode==="text";for(const row of data.rows){if(category!=="all"&&row.category!==category)continue;const card=document.createElement("article"),h=document.createElement("h3"),grid=document.createElement("div");h.textContent=row.name+" | 固定文字："+row.category;card.append(h);grid.className="grid";const suffix=mode+(cfg?"_cfg2":""),number=String(current).padStart(6,"0");const images=[["高清参考","inputs/"+row.id+"_reference.png",true],["MC目标","inputs/"+row.id+"_target.png"],["实验前·CFG1","generated/qwen_hd_000000_"+row.id+"_"+mode+"_s0.png"],["Qwen高清特征"+(cfg?"·CFG2":"·CFG1"),"generated/qwen_hd_"+number+"_"+row.id+"_"+suffix+"_s"+seed+".png"],["目标像素对照","generated/oracle_target_"+number+"_"+row.id+"_"+suffix+"_s"+seed+".png"],["Qwen无参考·类别文字","generated/qwen_hd_"+number+"_"+row.id+"_text_s"+seed+".png"]];for(const [label,path,hd] of images){const cell=document.createElement("div"),p=document.createElement("p"),img=document.createElement("img");p.textContent=label;img.src=path;img.alt=label;img.loading="lazy";if(hd)img.className="hd";cell.append(p,img);grid.append(cell)}card.append(grid);parent.append(card)}}for(const id of ["category","step","mode","seed","cfg"])document.getElementById(id).addEventListener("change",draw);for(const name of data.panels){const d=document.createElement("details"),s=document.createElement("summary"),a=document.createElement("a"),img=document.createElement("img");s.textContent=name;a.href="panels/"+name;img.src=a.href;img.loading="lazy";a.append(img);d.append(s,a);document.getElementById("panels").append(d)}draw();</script>'''
    (root/'index.html').write_text(page.replace('DATA',data))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_overfit_20261108'))
    parser.add_argument('--step',type=int,default=2000)
    parser.add_argument('--publish',type=Path,default=Path('outputs/native16_data_review_20261107/overfit'))
    args=parser.parse_args()
    main(args.out,args.step,args.publish)
