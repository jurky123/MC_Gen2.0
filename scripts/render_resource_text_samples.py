"""Review exact file-derived prompts used by the new training run."""
import json,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mcgen.images import content_hash
ROOT=Path('outputs/resource_text_training_20261103')
if __name__=='__main__':
 rows=json.load(open(ROOT/'sample.json'));out=ROOT/'review';out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);maps={}
 old={}
 for line in (ROOT/'decisions.jsonl').open():
  r=json.loads(line)
  if r['target_hash'] in {x['target_hash'] for x in rows}:old[r['target_hash']]=r['old_text']
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18);bold=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>纯文件文本训练抽样</title><style>body{font:17px system-ui;margin:30px;background:#f4f3ed;color:#23382f}article{display:grid;grid-template-columns:140px 1fr;gap:24px;padding:22px;margin:15px;background:white;border-radius:12px}img{width:128px;height:128px;image-rendering:pixelated;background:#808080}p{overflow-wrap:anywhere}</style><h1>纯文件文本：当前训练使用的标签</h1><p>24 张随机训练图片 + 8 张有技术标记清理的随机图片；原样展示实际 prompt，不含视觉描述。所有图片保持原像素。</p>']
 for i,r in enumerate(rows):
  s=r['target']
  if s['path'] not in maps:maps[s['path']]=np.memmap(s['path'],dtype='uint8',mode='r',shape=tuple(s['shape']))
  a=np.array(maps[s['path']][s['index']]);assert content_hash(a)==r['target_hash'];Image.fromarray(a).save(out/'images'/f'{i:02d}.png')
  e=html.escape;page.append(f'<article><div>#{i:02d}<br><img src="images/{i:02d}.png"></div><div><p>文件：{e(r["source_filename"])}</p><p>原抽取文本：{e(old[r["target_hash"]])}</p><p><b>当前训练文本：{e(r["prompt"])}</b></p><p>移除的技术标记：{e(str(r["removed_technical_tokens"]))}</p></div></article>')
 page.append('</html>');(out/'index.html').write_text(''.join(page));print('32 file-text review cards saved')
