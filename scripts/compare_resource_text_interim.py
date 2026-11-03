"""Snapshot in-flight checkpoints and compare unchanged prompt/seed generation."""
import argparse,datetime,html,json,shutil,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.runtime import load_checkpoint
from mcgen.model import PixelFlow
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from mcgen.images import decode,composite
from run_resource_text_training import merged
from diagnose_pixel_integrity import stats
from evaluate_text_recovery import color_fraction
RUN=Path('outputs/resource_text_training_20261103')

def main(device, continuation_reference=False, validate_flow=False):
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,device)
 out=Path('outputs')/('resource_text_comparison_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();(out/'checkpoints').mkdir()
 paths={'before':Path('outputs/fullscale_text10k_20261103/training/best_flow.pt'),'best':RUN/'training/best_flow.pt','latest':RUN/'training/latest.pt'}
 if continuation_reference:paths={'before':Path('outputs/fullscale_text10k_20261103/training/best_flow.pt'),'continued':Path('outputs/resource_text_training_20261103/training/best_flow.pt'),'best':RUN/'training/best_flow.pt'}
 protocol={'sampling':{'steps':32,'solver':'heun','text_scale':2,'precision':'float32'},'checkpoints':{}}
 for arm,source in paths.items():
  dest=out/'checkpoints'/f'{arm}.pt';shutil.copyfile(source,dest);state=load_checkpoint(dest)
  protocol['checkpoints'][arm]={'source':str(source),'step':state['step'],'sha256':file_hash(dest),'text_identity':state['text_identity']};del state
 cases=json.load(open(RUN/'eval_cases.json'));atomic_json(out/'cases.json',cases);atomic_json(out/'protocol.json',protocol)
 cfg=Experiment.load(RUN/'config.yaml');encoder=merged();assert all(r['text_identity']==encoder.identity for r in protocol['checkpoints'].values())
 if validate_flow:
  from mcgen.training import Trainer
  validator=Trainer(cfg,RUN/'catalog',out/'flow_validation',device,device,encoder=encoder)
  flow={}
  for arm in paths:
   state=load_checkpoint(out/'checkpoints'/f'{arm}.pt');validator.model.load_state_dict(state['model']);del state
   flow[arm]=validator.validate();print('flow',arm,flow[arm],flush=True)
  atomic_json(out/'flow_validation.json',flow);del validator;torch.cuda.empty_cache()
 model=PixelFlow(cfg.model).to(device);results=[]
 for arm in paths:
  state=load_checkpoint(out/'checkpoints'/f'{arm}.pt');model.load_state_dict(state['model']);model.eval();del state;(out/arm).mkdir()
  for i in range(0,len(cases),8):
   batch=cases[i:i+8];text,mask=encoder.encode(conditioned_prompts([r['evaluation_prompt'] for r in batch],cfg),device)
   with torch.inference_mode():raw=sample(model,text,mask,seeds=[r['seed'] for r in batch],steps=32,text_scale=2,solver='heun')
   for j,(r,value) in enumerate(zip(batch,raw)):
    image=decode(value);Image.fromarray(image).save(out/arm/f'{i+j:03d}.png')
    results.append({'arm':arm,'case':i+j,'group':r['group'],'pixel_stats':stats(image),'color_fraction':color_fraction(image,r['requested_color']) if r.get('requested_color') else None})
  print('evaluated',arm,protocol['checkpoints'][arm]['step'],flush=True)
 atomic_json(out/'results.json',results)
 if continuation_reference:
  names={'before':'旧版已认可模型','continued':f'上轮续训 · {protocol["checkpoints"]["continued"]["step"]} 步','best':f'从零训练 · {protocol["checkpoints"]["best"]["step"]} 步'}
 else:
  names={'before':'原 best（本轮起点）','best':f'本轮 best · {protocol["checkpoints"]["best"]["step"]} 步','latest':f'最新模型 · {protocol["checkpoints"]["latest"]["step"]} 步'}
 groups={}
 for i,r in enumerate(cases):
  key=r['category'] if r['group']=='novel_composition' else 'longtail'
  groups.setdefault(key,[]).append(i)
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19);bold=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
 for group,indices in groups.items():
  n=6 if group!='longtail' else 8
  for start in range(0,len(indices),n):
   batch=indices[start:start+n];sheet=Image.new('RGB',(1100,96+len(batch)*146),'#f5f4ef');d=ImageDraw.Draw(sheet)
   d.text((20,10),f'文生图对比 · {group} · 相同提示和种子 · 原始输出',font=bold,fill='#23382f')
   for col,arm in enumerate(paths):d.text((390+col*230,57),names[arm],font=font,fill='#23382f')
   for row,i in enumerate(batch):
    y=96+row*146;r=cases[i]
    d.multiline_text((20,y+18),'\n'.join(textwrap.wrap(r['evaluation_prompt'],32)),font=font,fill='#23382f');d.text((20,y+105),f'seed: {r["seed"]}',font=font,fill='#62756b')
    for col,arm in enumerate(paths):
     im=Image.open(out/arm/f'{i:03d}.png').convert('RGBA').resize((128,128),Image.Resampling.NEAREST);bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im);sheet.paste(bg.convert('RGB'),(390+col*230,y))
   sheet.save(out/f'{group}_{start//n+1}.png')
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>训练中生图对照</title><style>body{font:16px system-ui;margin:28px;background:#f5f4ef;color:#23382f}table{border-collapse:collapse;width:100%}td,th{padding:16px;border-bottom:1px solid #ccd4cc;text-align:center}th{position:sticky;top:0;background:#eef0e7}img{width:128px;height:128px;image-rendering:pixelated;background:#808080}</style><h1>纯文件文本训练：生图对照</h1><p>所有68个固定提示与种子，32步 Heun / CFG2。模型先快照再评估；图像只做最近邻放大，不调色、不修轮廓。</p><table><tr><th>提示 / 种子</th>']+[f'<th>{v}</th>' for v in names.values()]+['</tr>']
 for i,r in enumerate(cases):
  page.append(f'<tr><td>{html.escape(r["evaluation_prompt"])}<br>{r["seed"]}</td>');page.extend(f'<td><img src="{arm}/{i:03d}.png"></td>' for arm in paths);page.append('</tr>')
 page.append('</table></html>');(out/'index.html').write_text(''.join(page))
 summary={}
 for arm in paths:
  rows=[r for r in results if r['arm']==arm and r['group']=='novel_composition'];summary[arm]={'n':len(rows),'requested_color_fraction':float(np.mean([r['color_fraction'] for r in rows])),**{k:float(np.mean([r['pixel_stats'][k] for r in rows])) for k in rows[0]['pixel_stats']}}
 atomic_json(out/'summary.json',summary);atomic_json(RUN/'latest_interim_comparison.json',{'output':str(out),'protocol':protocol});print('OUTPUT',out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--run',type=Path,default=RUN);p.add_argument('--continuation-reference',action='store_true');p.add_argument('--validate-flow',action='store_true');a=p.parse_args();RUN=a.run;main(a.device,a.continuation_reference,a.validate_flow)
