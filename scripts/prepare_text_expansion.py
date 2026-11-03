"""Select new complete-item candidates for a roughly 2k-target mixed-caption run."""
import json,random,re,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw,ImageFont
from mcgen.catalog import read_catalog,atomic_json,file_hash
from mcgen.images import composite
from audit_item_captions import array

ROOT=Path('outputs/text_expansion2k_20261102')
CATEGORIES='pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot gem apple book bow'.split()
BAD=set('head blade handle binding broken cast casting blueprint guide template icon outline pattern part piece overlay empty slot form mold mould stamp seeds seed shard slice sandwich cookie pie juice jam ice cream tea compound brew base tconstruct tinkers tinker ticex construct constructs'.split())
if __name__=='__main__':
 if ROOT.exists():raise ValueError('Use a fresh experiment directory')
 records,meta=read_catalog('data/catalogs/native16_v3/catalog')
 names=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=['file_name','mod_slug']).to_pandas()
 used=set()
 for path in ['outputs/text_caption_scale1152_20261102/vision/cases.jsonl','outputs/semantic_caption_refined256_v3_20261102/cases.jsonl','outputs/text_caption_audit_20261102/cases.jsonl']:
  used.update(r['target_hash'] for r in map(json.loads,Path(path).read_text().splitlines()))
 pools={};seen=set()
 for r in records:
  if r['asset_type']!='item' or r['split'] not in ('train','val') or r['target_hash'] in used or r['target_hash'] in seen:continue
  seen.add(r['target_hash']);filename=str(names.iloc[r['source_row']]['file_name']);words=re.sub('[^a-z]+',' ',filename.lower()).split()
  category=next((c for c in CATEGORIES if c in words),None)
  if not category or set(words)&BAD:continue
  pools.setdefault((r['split'],category),[]).append({**r,'reference':None,'source_filename':filename,'source_mod':str(names.iloc[r['source_row']]['mod_slug']),'old_caption':r['prompt'],'category':category})
 rng=random.Random(20261114);chosen=[];projects=Counter()
 for values in pools.values():rng.shuffle(values)
 for split,total in [('train',1800),('val',192)]:
  selected=[]
  while len(selected)<total:
   changed=False
   for category in CATEGORIES:
    pool=pools.get((split,category),[])
    while pool:
     r=pool.pop()
     if projects[r['lineage']]>=8:continue
     selected.append(r);projects[r['lineage']]+=1;changed=True;break
    if len(selected)>=total:break
   if not changed:raise ValueError('Not enough candidates under source cap')
  chosen.extend(selected)
 chosen.sort(key=lambda r:(r['split'],CATEGORIES.index(r['category']),r['id']))
 ROOT.mkdir();(ROOT/'review').mkdir()
 target=ROOT/'candidate_targets.mmap';imgs=np.stack([array(r) for r in chosen]);imgs.tofile(target)
 for i,r in enumerate(chosen):r['review_index']=i;r['target']={'path':str(target.resolve()),'shape':list(imgs.shape),'index':i}
 atomic_json(ROOT/'candidates.json',chosen)
 atomic_json(ROOT/'selection_protocol.json',{'parent_fingerprint':meta['fingerprint'],'source_cap':8,'categories':CATEGORIES,'excluded_filename_tokens':sorted(BAD),'candidate_count':len(chosen),'candidate_sha256':file_hash(ROOT/'candidates.json'),'reuse_core':'Existing reviewed core training/validation examples will be retained separately.','test_used':False})
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',11)
 for start in range(0,len(chosen),96):
  sheet=Image.new('RGB',(1200,8*112),'white');draw=ImageDraw.Draw(sheet)
  for k,r in enumerate(chosen[start:start+96]):
   x=(k%12)*100;y=(k//12)*112
   draw.text((x+2,y),f'{r["review_index"]} {r["category"]}',font=font,fill='black')
   short=Path(r['source_filename']).stem
   draw.text((x+2,y+14),short[:15],font=font,fill='black')
   sheet.paste(Image.fromarray(composite(imgs[start+k],128).round().astype('uint8')).resize((64,64),Image.Resampling.NEAREST),(x+18,y+37))
  sheet.save(ROOT/'review'/f'page_{start//96:02d}.png')
 print('prepared',len(chosen),dict(Counter(r['category'] for r in chosen)))
