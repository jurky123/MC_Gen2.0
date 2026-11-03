"""Compare train-set fitting at initialization, best-flow step and final endpoint."""
import argparse,json,random,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.runtime import Runtime,load_checkpoint
from mcgen.evaluation import metrics,summarize
from mcgen.images import composite
from audit_item_captions import array
ROOT=Path('outputs/text_expansion2k_20261102');OUT=ROOT/'fit_diagnosis'

def prepare():
 rows=json.loads((ROOT/'cases.json').read_text());byid={r['id']:r for r in rows};counts=Counter(r['short'] for r in rows if r['split']=='train');rng=random.Random(20261118)
 new=[r for r in rows if r['split']=='train' and r['caption_origin']=='expansion_automatic_weak_label' and r['long_caption_eligible'] and counts[r['short']]==1];rng.shuffle(new)
 picked=[]
 for category in sorted({r['category'] for r in new}):
  group=[r for r in new if r['category']==category];assert len(group)>=2;picked.extend(group[:2])
 evalcases=json.loads((ROOT/'eval_cases.json').read_text())['cases'];old=[byid[r['id']] for r in evalcases if r['group']=='train_fit_only'];val=[byid[r['id']] for r in evalcases if r['group']=='expansion_val_short']
 cases=[]
 for group,rs in [('new_train',picked),('old_train',old),('new_val',val),('filename_only_train',[r for r in rows if r['split']=='train' and not r['long_caption_eligible']])]:
  for r in rs:
   for kind in r['caption_variants']:cases.append({**r,'evaluation_prompt':r[kind],'group':group+'_'+kind,'seed':20261118})
 OUT.mkdir(exist_ok=False);(OUT/'images').mkdir()
 paths={'before':Path('outputs/text_recovery_core_20261102/training/latest.pt'),'best_flow':ROOT/'training/best_flow.pt','final':ROOT/'training/latest.pt'}
 atomic_json(OUT/'protocol.json',{'cases':cases,'selection_seed':20261118,'selection':'Two new training images per category with unambiguous filename labels; all 16 prior training probes; same 28 new validation targets; all 9 filename-only training targets. Chosen before diagnostic generation, after original expansion evaluation.','checkpoints':{k:{'path':str(p),'sha256':file_hash(p)} for k,p in paths.items()},'steps':32,'solver':'heun','cfg':2,'purpose':'Post-hoc fitting diagnosis, not a new unbiased benchmark or automatic checkpoint selection.'})
 return cases,paths

def run(cases=None,paths=None):
 torch.set_num_threads(4)
 if cases is None:cases,paths=prepare()
 device='cuda:1'
 while torch.cuda.mem_get_info(device)[0]<23*1024**3:time.sleep(30)
 free,total=torch.cuda.mem_get_info(device);torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),device)
 runtime=Runtime(paths['before'],device,device);records=[];all_images={};steps={}
 for arm,path in paths.items():
  state=load_checkpoint(path);assert state['text_identity']==runtime.encoder.identity;steps[arm]=state['step'];runtime.model.load_state_dict(state['model']);del state
  images=[]
  for start in range(0,len(cases),8):
   batch=cases[start:start+8];images.extend(runtime.generate([r['evaluation_prompt'] for r in batch],seeds=[r['seed'] for r in batch],steps=32,solver='heun',text_scale=2));atomic_json(OUT/'status.json',{'stage':'generating','arm':arm,'done':len(images),'total':len(cases)})
  for i,(r,img) in enumerate(zip(cases,images)):
   Image.fromarray(img).save(OUT/'images'/f'{arm}_{i:03d}.png');score=metrics(img,array(r));score['seam_mae']=None;score['no_foreground']=float(not (img[...,3]>=128).any());records.append({'id':r['id'],'mode':arm+':'+r['group'],'lineage':r['lineage'],'seed':r['seed'],'metrics':score})
  all_images[arm]=images;print('generated',arm,len(images),flush=True)
 (OUT/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));atomic_json(OUT/'summary.json',summarize(records))
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
 for group in dict.fromkeys(r['group'] for r in cases):
  indices=[i for i,r in enumerate(cases) if r['group']==group]
  for start in range(0,len(indices),6):
   canvas=Image.new('RGB',(1080,30+6*150),'white');d=ImageDraw.Draw(canvas);d.text((8,3),f'{group} | TARGET | '+ ' | '.join(a.upper() for a in paths),font=font,fill='black')
   for j,i in enumerate(indices[start:start+6]):
    r=cases[i];y=30+j*150;lines=[];line=''
    for ch in r['evaluation_prompt']:
     if font.getlength(line+ch)>480:lines.append(line);line=ch
     else:line+=ch
    lines.append(line);d.multiline_text((8,y),'\n'.join(lines),font=font,fill='black')
    for col,img in enumerate([array(r)]+[all_images[a][i] for a in paths]):canvas.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((128,128),Image.Resampling.NEAREST),(520+col*138,y))
   canvas.save(OUT/f'{group}_{start//6:02d}.png')
 atomic_json(OUT/'status.json',{'stage':'complete','cases':len(cases),'checkpoint_steps':steps})
 for group in dict.fromkeys(r['group'] for r in cases):
  print(group,flush=True)
  for arm in paths:
   rs=[r for r in records if r['mode']==arm+':'+group];print(arm,{k:round(float(np.mean([r['metrics'][k] for r in rs])),4) for k in ['composite_mae','alpha_iou']},flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--extended-root',type=Path);args=parser.parse_args()
 if args.extended_root:
  cases=json.loads((OUT/'protocol.json').read_text())['cases'];OUT=args.extended_root/'fit_diagnosis';OUT.mkdir(exist_ok=False);(OUT/'images').mkdir()
  paths={'before':ROOT/'training/latest.pt','best_flow':args.extended_root/'training/best_flow.pt','final':args.extended_root/'training/latest.pt'}
  atomic_json(OUT/'protocol.json',{'cases':cases,'checkpoints':{k:{'path':str(p),'sha256':file_hash(p)} for k,p in paths.items()},'note':'Same frozen 157 fitting probes and seeds as preceding diagnosis, comparing expansion step 6000, minimum validation-flow checkpoint and extended endpoint. All generated afresh in matching batches.','steps':32,'solver':'heun','cfg':2})
  run(cases,paths)
 else:run()
