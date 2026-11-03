"""Prepare reviewable complete-item candidates without changing canonical splits."""
import json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mcgen.images import composite
from mcgen.catalog import atomic_json
from audit_item_captions import array

ROOT=Path('outputs/text_recovery_core_20261102')
SOURCE=Path('outputs/text_caption_scale1152_20261102')
CATEGORIES='pickaxe sword shovel axe hoe helmet chestplate leggings boots shield ingot gem apple book bow'.split()
if __name__=='__main__':
 ROOT.mkdir(exist_ok=False);(ROOT/'review').mkdir()
 cases={r['id']:r for r in map(json.loads,(SOURCE/'vision/cases.jsonl').read_text().splitlines())}
 labels=[json.loads(s) for s in (SOURCE/'refined/semantic/candidates.jsonl').read_text().splitlines()]
 rejected={r['id'] for r in json.loads((SOURCE/'quality_review.json').read_text())['ratings'] if r['new_rating']!='plausible'}
 rows=[]
 for label in labels:
  c=cases[label['id']];words=re.sub('[^a-z]+',' ',c['source_filename'].lower()).split()
  category=next((k for k in CATEGORIES if k in words),None)
  if not category or set(words)&set('head blade handle binding broken cast casting blueprint guide template icon outline pattern part piece'.split()):continue
  if category not in label['combined_text'].lower() or c['id'] in rejected or set(label['quality_flags'])&{'name_conflict','review_schema_error','background_in_caption'}:continue
  rows.append({**c,'reference':None,'category':category,'long':label['combined_text'],'annotation_flags':label['quality_flags']})
 rows.sort(key=lambda r:(r['split'],CATEGORIES.index(r['category']),r['id']))
 for i,r in enumerate(rows):r['review_index']=i
 atomic_json(ROOT/'candidates.json',rows)
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',12)
 for start in range(0,len(rows),48):
  sheet=Image.new('RGB',(1200,8*142),'white');d=ImageDraw.Draw(sheet)
  for k,r in enumerate(rows[start:start+48]):
   x=(k%6)*200;y=(k//6)*142
   d.text((x+3,y),f'{r["review_index"]} {r["split"]} {r["category"]}',fill='black',font=font)
   name=Path(r['source_filename']).stem
   d.text((x+3,y+16),name[:28],fill='black',font=font)
   sheet.paste(Image.fromarray(composite(array(r),128).round().astype('uint8')).resize((96,96),Image.Resampling.NEAREST),(x+50,y+38))
  sheet.save(ROOT/'review'/f'page_{start//48:02d}.png')
 print('candidates',len(rows))
