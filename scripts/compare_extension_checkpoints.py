"""Direct final-vs-best paired statistics and complete three-checkpoint image index."""
import html,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mcgen.catalog import atomic_json
from mcgen.images import composite
ROOT=Path('outputs/text_expansion22k_conservative_20261102')
if __name__=='__main__':
 cases=json.loads((ROOT/'eval_cases.json').read_text())['cases'];paths={'before':ROOT/'evaluation','best_flow':ROOT/'evaluation_best_flow','final':ROOT/'evaluation'}
 rows={}
 for arm,path in paths.items():
  prefix='before:' if arm=='before' else 'recovery:'
  rows[arm]={(r['id'],r['seed'],r['mode'].split(':',1)[1]):r for r in map(json.loads,(path/'metrics.jsonl').read_text().splitlines()) if r['mode'].startswith(prefix)}
 assert rows['before'].keys()==rows['best_flow'].keys()==rows['final'].keys()
 equal=[]
 for i in range(len(cases)):
  a=np.asarray(Image.open(paths['before']/'images'/f'before_{i:03d}.png'));b=np.asarray(Image.open(paths['best_flow']/'images'/f'before_{i:03d}.png'));equal.append(np.array_equal(a,b))
 result={};rng=np.random.default_rng(20261120)
 for group in dict.fromkeys(r['group'] for r in cases):
  keys=[key for key in rows['final'] if key[2]==group];scores={}
  for metric in ['composite_mae','alpha_iou','no_foreground','requested_color_fraction']:
   if metric not in rows['final'][keys[0]]['metrics']:continue
   byproject=defaultdict(list)
   for key in keys:
    b,f=rows['best_flow'][key],rows['final'][key];byproject[f['lineage']].append(f['metrics'][metric]-b['metrics'][metric])
   values=np.array([np.mean(v) for v in byproject.values()]);ci=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975])
   scores[metric]={'sample_means':{a:float(np.mean([rs[k]['metrics'][metric] for k in keys])) for a,rs in rows.items()},'final_minus_best_project_mean':float(values.mean()),'paired_project_bootstrap_95':ci.tolist()}
  result[group]={'n':len(keys),'metrics':scores}
 best_step=json.loads((ROOT/'evaluation_best_flow/protocol.json').read_text())['endpoint']
 atomic_json(ROOT/'checkpoint_comparison.json',{'best_flow_step':best_step,'final_step':22000,'baseline_images_identical_between_evaluations':sum(equal),'total_baseline_images':len(equal),'groups':result,'limitation':'Single training run and development datasets; best checkpoint selected by validation flow loss, not generation benchmark. No automatic promotion.'})
 parts=['<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:25px}td{padding:10px;border:1px solid #ccc}img{width:128px;image-rendering:pixelated;background:#808080}table{border-collapse:collapse}</style><h1>6000 / 最低验证误差 / 22000 步</h1><p>同一提示、同一随机种子、32 步 Heun、CFG 2。完整展示冻结案例。</p><table><tr><th>提示</th><th>6000 步</th><th>最佳验证 '+str(best_step)+' 步</th><th>最终 22000 步</th></tr>']
 for i,r in enumerate(cases):parts.append(f'<tr><td>{html.escape(r["evaluation_prompt"])}<br>{html.escape(r["group"])} / {r["seed"]}</td><td><img src="evaluation/images/before_{i:03d}.png"></td><td><img src="evaluation_best_flow/images/recovery_{i:03d}.png"></td><td><img src="evaluation/images/recovery_{i:03d}.png"></td></tr>')
 parts.append('</table>');(ROOT/'checkpoint_comparison.html').write_text('\n'.join(parts))
 groups=defaultdict(list)
 for i,r in enumerate(cases):
  if r['group']=='novel_composition':groups[r['evaluation_prompt']].append(i)
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',13);canvas=Image.new('RGB',(1260,40+6*112),'white');draw=ImageDraw.Draw(canvas);draw.text((6,7),f'Each prompt: 6000 (2 seeds) | best {best_step} (2 seeds) | final 22000 (2 seeds)',font=font,fill='black')
 for n,(prompt,indices) in enumerate(groups.items()):
  x=n%3*420;y=40+n//3*112;draw.text((x+4,y),prompt,font=font,fill='black')
  for j,(arm,i) in enumerate((arm,i) for arm in paths for i in indices):
   prefix='before' if arm=='before' else 'recovery';img=np.asarray(Image.open(paths[arm]/'images'/f'{prefix}_{i:03d}.png'))
   canvas.paste(Image.fromarray(composite(img,128).round().astype('uint8')).resize((64,64),Image.Resampling.NEAREST),(x+4+j*68,y+30))
 canvas.save(ROOT/'checkpoint_comparison_basic.png');print('best',best_step,'final',22000,'baseline equality',sum(equal),len(equal))
