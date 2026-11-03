"""Frozen recovery probes: development holdout, broad regression and novel prompts."""
import argparse,gc,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.runtime import Runtime,load_checkpoint
from mcgen.evaluation import metrics,summarize
from mcgen.images import composite
from audit_item_captions import array

ROOT=Path('outputs/text_recovery_core_20261102')
BASE=Path('outputs/text_caption_scale1152_20261102')

def prepare():
 dest=ROOT/'eval_cases.json'
 if dest.exists():raise ValueError('Evaluation cases already frozen')
 rows=json.loads((ROOT/'cases.json').read_text());cases=[]
 for r in rows:
  if r['split']=='val':cases.append({**r,'evaluation_prompt':r['short'],'group':'core_val','seed':20261112})
 rng=random.Random(20261112);train=[r for r in rows if r['split']=='train'];rng.shuffle(train)
 chosen=[]
 for category in sorted({r['category'] for r in train}):chosen.append(next(r for r in train if r['category']==category))
 chosen.extend(r for r in train if r['id'] not in {x['id'] for x in chosen})
 for r in chosen[:16]:cases.append({**r,'evaluation_prompt':r['short'],'group':'train_fit_only','seed':20261112})
 broad={r['id']:r for r in map(json.loads,(BASE/'vision/cases.jsonl').read_text().splitlines())}
 for c in json.loads((BASE/'evaluation/protocol.json').read_text())['cases']:
  r=broad[c['id']];assert r['split']=='val' and r['target_hash']==c['target_hash']
  cases.append({**r,'evaluation_prompt':c['prompt'],'group':'broad_'+c['kind'],'seed':c['seed']})
 for category,colors in [('sword',['red','blue','green']),('gem',['red','blue','purple']),('helmet',['golden','blue','green']),('apple',['red','blue','golden']),('pickaxe',['red','blue','purple']),('shield',['red','blue','green'])]:
  for color in colors:
   for seed in [20261112,20261113]:
    cases.append({'id':f'novel:{color}:{category}','evaluation_prompt':f'a {color} {category}','group':'novel_composition','category':category,'requested_color':color,'seed':seed,'target':None,'lineage':'synthetic:'+category})
 atomic_json(dest,{'cases':cases,'steps':32,'solver':'heun','text_scale':2,'note':'Frozen before recovery generation. Novel prompts have no pixel target. Core and broad validation overlap; do not pool their scores. Development evaluation, not locked test.'})
 print('frozen',len(cases))


def color_fraction(img,color):
 a=img.astype(float)/255;visible=a[...,3]>=.5
 if not visible.any():return 0.
 from colorsys import rgb_to_hsv
 hsv=np.array([rgb_to_hsv(*p) for p in a[...,:3][visible]])
 h,s,v=hsv.T;valid=(s>.25)&(v>.12)
 ranges={'red':(h<.045)|(h>.95),'blue':(h>=.52)&(h<.72),'green':(h>=.20)&(h<.48),'purple':(h>=.72)&(h<.92),'golden':(h>=.08)&(h<.20)}
 return float((ranges[color]&valid).mean())


def run(device,balanced=False,best_flow=False):
 if balanced and best_flow:raise ValueError("Choose either balanced or best-flow")
 out=ROOT/('evaluation_balanced' if balanced else ('evaluation_best_flow' if best_flow else 'evaluation'));out.mkdir(exist_ok=False);(out/'images').mkdir()
 checkpoint=ROOT/('balanced_candidate.pt' if balanced else ('training/best_flow.pt' if best_flow else 'training/latest.pt'))
 protocol=json.loads((ROOT/'protocol.json').read_text());cases=json.loads((ROOT/'eval_cases.json').read_text())['cases']
 while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
 runtime=Runtime(protocol['initialization'],device,device);all_rows=[];all_images={}
 for arm in ['before','recovery']:
  if arm=='recovery':
   state=load_checkpoint(checkpoint)
   if balanced:
    assert state['inference_only'] and state['weight_interpolation']['recovery_weight']==.5
    assert state['weight_interpolation']['parents'][0]['sha256']==protocol['initialization_sha256']
    assert state['weight_interpolation']['parents'][1]['sha256']==file_hash(ROOT/'training/latest.pt')
   elif best_flow:
    scores=[r for r in map(json.loads,(ROOT/'training/metrics.jsonl').read_text().splitlines()) if 'text_flow_mse' in r]
    chosen=min(scores,key=lambda r:r['text_flow_mse'])
    assert state['step']==chosen['step'] and state['initialization']['sha256']==protocol['initialization_sha256']
   else:
    assert state['step']==protocol['endpoint'] and state['initialization']['sha256']==protocol['initialization_sha256']
   assert state['config']['model']==runtime.config.to_dict()['model'] and state['text_identity']==runtime.encoder.identity
   checkpoint_step=state['step'];runtime.model.load_state_dict(state['model']);del state
  images=[]
  for start in range(0,len(cases),8):
   batch=cases[start:start+8]
   images.extend(runtime.generate([r['evaluation_prompt'] for r in batch],seeds=[r['seed'] for r in batch],steps=32,solver='heun',text_scale=2))
   print(json.dumps({'arm':arm,'generated':len(images),'total':len(cases)}),flush=True)
  for i,(r,img) in enumerate(zip(cases,images)):
   Image.fromarray(img).save(out/'images'/f'{arm}_{i:03d}.png')
   score=metrics(img,array(r)) if r['target'] else {}
   if r['target']:score['seam_mae']=None
   else:score['requested_color_fraction']=color_fraction(img,r['requested_color'])
   score['no_foreground']=float(not (img[...,3]>=128).any());score['foreground_pixels']=int((img[...,3]>=128).sum())
   all_rows.append({'id':r['id'],'lineage':r['lineage'],'mode':arm+':'+r['group'],'seed':r['seed'],'metrics':score})
  all_images[arm]=images
  uncond=runtime.generate(['']*16,seeds=list(range(100,116)),steps=32,solver='heun',text_scale=0)
  grid=Image.new('RGB',(512,512),(128,128,128))
  for i,img in enumerate(uncond):grid.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((128,128),Image.Resampling.NEAREST),((i%4)*128,(i//4)*128))
  grid.save(out/f'unconditional_{arm}.png')
 (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows));atomic_json(out/'summary.json',summarize(all_rows))
 atomic_json(out/'protocol.json',{'eval_cases_sha256':file_hash(ROOT/'eval_cases.json'),'before_sha256':protocol['initialization_sha256'],'recovery_sha256':file_hash(checkpoint),'candidate_kind':'fixed_50_percent_parameter_blend' if balanced else ('minimum_validation_flow' if best_flow else 'trained_endpoint'),'checkpoint':str(checkpoint),'steps':32,'cfg':2,'endpoint':None if balanced else checkpoint_step,'color_metric':'HSV fraction of alpha>=128 pixels; diagnostic only, not shape or semantic accuracy.'})
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
 for group in dict.fromkeys(r['group'] for r in cases):
  indices=[i for i,r in enumerate(cases) if r['group']==group]
  for start in range(0,len(indices),8):
   sheet=Image.new('RGB',(1000,8*148+34),'white');draw=ImageDraw.Draw(sheet)
   draw.text((8,4),f'{group} | Target (if present) | Before | '+('50% blend' if balanced else 'After training'),font=font,fill='black')
   for k,i in enumerate(indices[start:start+8]):
    r=cases[i];y=34+k*148;lines=[];line=''
    for char in r['evaluation_prompt']:
     if font.getlength(line+char)>400:lines.append(line);line=char
     else:line+=char
    lines.append(line);draw.multiline_text((8,y),'\n'.join(lines)+f'\nseed {r["seed"]}',font=font,fill='black')
    columns=[(640,all_images['before'][i]),(820,all_images['recovery'][i])]
    if r['target']:columns.insert(0,(460,array(r)))
    for x,img in columns:sheet.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((128,128),Image.Resampling.NEAREST),(x,y))
   sheet.save(out/f'{group}_{start//8:02d}.png')
 print('evaluation complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--balanced',action='store_true');p.add_argument('--best-flow',action='store_true');p.add_argument('--device',default='cuda:1');p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();ROOT=a.root;torch.set_num_threads(4)
 if a.prepare:prepare()
 else:run(a.device,a.balanced,a.best_flow)
