"""Reproducible fresh samples of current labels, rendered verbatim for inspection."""
import csv,html,json,random,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from mcgen.images import content_hash
from mcgen.catalog import atomic_json,file_hash
SOURCE=Path('outputs/grounded_label_revision_10k_20261103/corrected_labels.jsonl')
OUT=Path('outputs/current_label_samples_20261103')
REASONS={'resource_component':'原路径表明是组件','possible_component':'文件名疑似组件','resource_layer':'原路径表明是图层','verified_emissive_layer':'确认是发光层','insufficient_visible_content':'可见内容过少','visual_identity_unresolved':'视觉身份或完整性不确定','annotation_process_in_caption':'描述混入背景或展示过程','visual_filename_identity_conflict':'视觉类别与文件名冲突','filename_identity_unverified':'文件语义未通过核验'}

def main():
 OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
 old={r['target_hash'] for r in json.load(open('outputs/training_label_audit_20261103/sample.json'))}
 pools={'dual':[],'visual':[],'quarantine':[]}
 for line in SOURCE.open():
  r=json.loads(line)
  if r['target_hash'] in old:continue
  g='quarantine' if r['decision']=='quarantine' else ('dual' if r['file_label'] else 'visual')
  pools[g].append(r)
 rng=random.Random(2026110307);selected=[]
 for g,n in [('dual',16),('visual',8),('quarantine',8)]:
  selected.extend({**r,'sample_group':g} for r in rng.sample(pools[g],n))
 names={'dual':'双标签候选','visual':'仅视觉标签候选','quarantine':'隔离，不用于训练'}
 maps={};font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18);bold=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
 def wrap(text,width=580):
  lines=[];line=''
  for word in text.split(' '):
   proposed=(line+' '+word).strip()
   if font.getlength(proposed)<=width:line=proposed;continue
   if line:lines.append(line);line=''
   for char in word:
    if font.getlength(line+char)>width:lines.append(line);line=char
    else:line+=char
  if line:lines.append(line)
  return lines

 for i,r in enumerate(selected):
  s=r['target']
  if s['path'] not in maps:maps[s['path']]=np.memmap(s['path'],mode='r',dtype='uint8',shape=tuple(s['shape']))
  a=np.array(maps[s['path']][s['index']]);assert content_hash(a)==r['target_hash']
  Image.fromarray(a).save(OUT/'images'/f'{i:02d}.png');r['review_id']=i;r['review_image']=f'images/{i:02d}.png'
 def labels(r):
  result=[('原文件名',r['source_filename']),('文件标签',r['file_label'] or '—（不使用）'),('视觉标签',r['visual_label'] or '—（不使用）')]
  if r['sample_group']=='quarantine':
   result.append(('隔离原因','；'.join(REASONS.get(f,f) for f in r['quality_flags'])))
   if r.get('visual_observation'):result.append(('观察记录（不用于训练）',r['visual_observation']['caption']))
  elif not r['file_label']:result.append(('文件标签说明','未通过语义核验，仅保留独立视觉描述'))
  return result
 for start in range(0,len(selected),4):
  batch=selected[start:start+4];heights=[];wrapped=[]
  for r in batch:
   lines=[]
   for name,text in labels(r):lines+=wrap(name+'：'+text)+['']
   wrapped.append(lines);heights.append(max(220,70+len(lines)*25))
  rowheights=[max(heights[0:2]),max(heights[2:4])];sheet=Image.new('RGB',(1600,95+sum(rowheights)),'#f4f3ed');d=ImageDraw.Draw(sheet)
  d.text((24,14),'修订后标签抽样 · '+names[batch[0]['sample_group']],font=bold,fill='#23382f');d.text((24,49),'全新固定种子抽样 · 标签原样展示 · 灰底表示透明区域 · 自动候选仍需人工检查',font=font,fill='#62756b')
  for j,r in enumerate(batch):
   x=(j%2)*800;y=90+(rowheights[0] if j>=2 else 0);h=rowheights[j//2]
   d.rounded_rectangle((x+10,y+5,x+790,y+h-8),12,fill='white')
   im=Image.open(OUT/r['review_image']).convert('RGBA').resize((128,128),Image.Resampling.NEAREST);bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im);sheet.paste(bg.convert('RGB'),(x+25,y+54));d.text((x+25,y+14),f'#{r["review_id"]:02d}',font=bold,fill='#23382f')
   yy=y+16
   for line in wrapped[j]:
    if line:d.text((x+180,yy),line,font=font,fill='#23382f')
    yy+=25
   assert yy<=y+h
  sheet.save(OUT/f'samples_{start//4+1:02d}.png')
 esc=lambda v:html.escape(str(v if v is not None else '—'))
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>当前修订标签抽样</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f4f3ed;color:#23382f}article{display:grid;grid-template-columns:140px 1fr;gap:22px;background:white;padding:24px;margin:18px 0;border-radius:12px}img{width:128px;height:128px;image-rendering:pixelated;background:#808080}p{line-height:1.6;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#267054}@media(max-width:600px){article{grid-template-columns:1fr}}</style><h1>当前修订标签：32 张全新抽样</h1><p>16 张双标签候选、8 张仅视觉标签候选、8 张隔离样本。固定种子随机抽取，并排除之前展示的72张。分组数量用于方便对照，不代表全库比例；没有按好坏挑选或改写描述。</p><p><a href="labels.csv">标签 CSV</a> · <a href="sample.json">完整记录</a></p>']
 for r in selected:
  page.append(f'<article><div><h3>#{r["review_id"]:02d}</h3><img src="{r["review_image"]}"><p>{names[r["sample_group"]]}</p></div><div>')
  for name,value in labels(r):page.append(f'<p><b>{esc(name)}</b><br>{esc(value)}</p>')
  page.append(f'<details><summary>原标签与来源</summary><p>原文件标签：{esc(r["old_filename_label"])}</p><p>原自动描述：{esc(r["old_visual_label"])}</p><p>原路径：{esc(r["resource_paths"])}</p><p>Mod：{esc(r["source_mod"])}</p><p>哈希：{r["target_hash"]}</p></details></div></article>')
 page.append('</html>');(OUT/'index.html').write_text(''.join(page))
 atomic_json(OUT/'sample.json',selected);atomic_json(OUT/'protocol.json',{'seed':2026110307,'source_sha256':file_hash(SOURCE),'exclude_prior_audit72':True,'pool_sizes':{k:len(v) for k,v in pools.items()},'counts':{'dual':16,'visual':8,'quarantine':8},'text_policy':'Exact current labels; no new captions or translations.'})
 with (OUT/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['review_id','sample_group','source_filename','file_label','visual_label','decision','quality_flags','target_hash'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(selected)
 with zipfile.ZipFile(OUT/'samples.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   if f.is_file() and f.suffix!='.zip':z.write(f,f.relative_to(OUT))
 print('Rendered32 fresh cases,8 sheets,HTML,CSV and ZIP.')
if __name__=='__main__':main()
