"""Export a tiny, attributed MIT texture subset for inspecting current file-text rules."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mcgen.catalog import atomic_json,file_hash
from mcgen.images import content_hash
from mcgen.resource_text import extract_resource_text
OUT=Path('examples/data_review')
NAMES=['molten_diamond_bucket.png','molten_brass_bucket.png','molten_copper_bucket.png','molten_gold_bucket.png','molten_iron_bucket.png','molten_netherite_bucket.png','molten_zinc_bucket.png','spruce_sword_blade.png','acacia_sword_guard.png','diamond_tool_grip.png','bamboo_sword_blade.png','acacia_sword_pommel.png','black_wool_tool_wrap.png','blue_wool_tool_wrap.png']
ISSUES={'acacia_sword_pommel.png':'剑柄尾部组件仍被保留：当前规则漏收。','black_wool_tool_wrap.png':'工具缠绕层仍被保留：当前规则漏收。','blue_wool_tool_wrap.png':'工具缠绕层仍被保留：当前规则漏收。'}
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
 a=json.load(open('outputs/training_label_audit_20261103/archive_audit.json'));archive=next(r for r in a if r['source_filename']=='spruce_sword_blade.png')
 selected={}
 for line in open('outputs/fullscale_text10k_20261103/catalog/samples.jsonl'):
  r=json.loads(line)
  if r['split']=='train' and r['caption_variant']=='short' and r.get('source_mod')=='create-modular-tools' and r['source_filename'] in NAMES:selected.setdefault(r['source_filename'],r)
 assert set(selected)==set(NAMES)
 rows=[];maps={}
 with zipfile.ZipFile(archive['archive_file']) as z:
  license_text=z.read('LICENSE_create_modular_tools').decode().replace('\r\n','\n');assert 'MIT License' in license_text and 'Copyright (c) 2024 ZLT' in license_text
  (OUT/'LICENSE.assets.txt').write_text(license_text)
  for i,name in enumerate(NAMES):
   r=selected[name];s=r['target']
   if s['path'] not in maps:maps[s['path']]=np.memmap(s['path'],dtype='uint8',mode='r',shape=tuple(s['shape']))
   pixels=np.array(maps[s['path']][s['index']]);assert content_hash(pixels)==r['target_hash']
   paths=[p for p in z.namelist() if p.endswith('/'+name) and content_hash(Image.open(io.BytesIO(z.read(p))).convert('RGBA'))==r['target_hash']]
   assert paths
   image=f'images/{i:02d}_{name}';Image.fromarray(pixels).save(OUT/image)
   info={'filename':name,'mod':'create-modular-tools','paths':paths,'visible_pixels':r['visible_pixels']}
   actual=extract_resource_text(**info)
   rows.append({'id':f'review-{i:02d}','image':image,'image_sha256':file_hash(OUT/image),'target_hash':r['target_hash'],'file_information':info,'old_filename_text':r['prompt'],'recorded_result':actual,'human_review':{'status':'known_component_omission' if name in ISSUES else 'review_example','note':ISSUES.get(name,'用于查看当前保留/隔离行为；不是人工精标。')},'source':{'project_id':'11TIcYqC','mod':'create-modular-tools','author':'ZLT','version_url':archive['url'],'archive_sha256':archive['archive_sha256'],'license':'MIT','license_file':'LICENSE.assets.txt'}})
 atomic_json(OUT/'samples.json',rows)
 cases=[]
 for filename in ['sun_gem.png','soul_crystal.png','glowing_sapphire_hoe.png','rabbage.png','TitanBase_sword.png','jade_gem.png','cactus_bow_pulling_0.png','oak_bow_pulling_2_spruce_arrow.png','gem4.png','item.sun_gem.png','body_tinkers_thinking_burnt_ashes.png','sensing_spell_book_e.png']:
  cases.append({'filename':filename,'recorded_result':extract_resource_text(filename),'kind':'synthetic_filename_only_no_corresponding_image'})
 atomic_json(OUT/'text_cases.json',cases)
 atomic_json(OUT/'manifest.json',{'purpose':'Small deterministic processing review set; not a representative dataset or benchmark','sampling':'Handpicked one verified MIT-licensed project;7 kept bucket textures,4 rejected parts,3 known missed parts. Filename-only synthetic cases are separately marked.','assets':14,'source_archive_sha256':archive['archive_sha256'],'policy_version':'resource_text_v1','policy_sha256':file_hash('src/mcgen/resource_text.py'),'current_training':'Scratch generator, file-information supervision only. Images/labels here reflect the current v1 rules, including disclosed omissions.'})
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18);bold=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
 sheet=Image.new('RGB',(1500,75+7*160),'#f4f3ed');d=ImageDraw.Draw(sheet);d.text((20,15),'数据处理检查样本 · 包含已知漏收例子 · 原图仅最近邻放大',font=bold,fill='#23382f')
 for i,r in enumerate(rows):
  x=(i%2)*750;y=75+(i//2)*160;im=Image.open(OUT/r['image']).convert('RGBA').resize((96,96),Image.Resampling.NEAREST);bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im);sheet.paste(bg.convert('RGB'),(x+20,y+10))
  result=r['recorded_result'];texts=[r['file_information']['filename'],'文本：'+(result['text'] or '—'),'结果：'+result['decision']]
  if r['human_review']['status'].startswith('known'):texts.append('已知问题：组件漏收')
  for j,text in enumerate(texts):d.text((x+135,y+8+j*29),text,font=font,fill='#23382f')
 sheet.save(OUT/'preview.png');print('14 attributed assets +12 synthetic filename examples exported.')
