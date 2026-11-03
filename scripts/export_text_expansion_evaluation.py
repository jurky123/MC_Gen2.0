"""Make all fixed basic prompts visible, with every seed and both checkpoints."""
import html,json,argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mcgen.images import composite
ROOT=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);ROOT=parser.parse_args().root
 cases=json.loads((ROOT/'eval_cases.json').read_text())['cases'];out=ROOT/'evaluation';groups=defaultdict(list)
 for i,r in enumerate(cases):
  if r['group']=='novel_composition':groups[r['evaluation_prompt']].append(i)
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
 sheet=Image.new('RGB',(1080,40+((len(groups)+2)//3)*122),'white');d=ImageDraw.Draw(sheet);d.text((8,8),'Each prompt: BEFORE seed 1 / seed 2 | AFTER seed 1 / seed 2',font=font,fill='black')
 for n,(prompt,indices) in enumerate(groups.items()):
  assert len(indices)==2
  x=n%3*360;y=40+n//3*122;d.text((x+4,y),prompt,font=font,fill='black')
  for j,(arm,i) in enumerate((arm,i) for arm in ['before','recovery'] for i in indices):
   img=np.asarray(Image.open(out/'images'/f'{arm}_{i:03d}.png').convert('RGBA'));sheet.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((80,80),Image.Resampling.NEAREST),(x+4+j*86,y+30))
 sheet.save(out/'all_basic_prompts.png')
 parts=['<!doctype html><meta charset="utf-8"><title>Expansion comparison</title><style>body{font:16px system-ui;margin:30px}td{padding:10px;border:1px solid #ccc}img{width:128px;image-rendering:pixelated;background:#808080}table{border-collapse:collapse}</style><h1>固定提示：扩容前 / 扩容后</h1><p>所有案例按冻结顺序展示；每行使用相同 seed。图片未作轮廓或颜色修正。</p><table><tr><th>提示</th><th>扩容前</th><th>扩容后</th></tr>']
 for i,r in enumerate(cases):parts.append(f'<tr><td>{html.escape(r["evaluation_prompt"])}<br><small>{html.escape(r["group"])} / seed {r["seed"]}</small></td><td><img src="images/before_{i:03d}.png"></td><td><img src="images/recovery_{i:03d}.png"></td></tr>')
 parts.append('</table>');(out/'comparison.html').write_text('\n'.join(parts));print('Exported',len(cases),'paired cases')
