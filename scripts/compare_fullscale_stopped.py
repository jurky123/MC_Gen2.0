"""Three-way report for the stopped 6000-update full-scale experiment."""
from collections import defaultdict
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from mcgen.catalog import atomic_json, file_hash

ROOT=Path('outputs/fullscale_eval6000_20261103')


def main():
    cases=json.load(open(ROOT/'eval_cases.json'))['cases']
    metrics=defaultdict(lambda:defaultdict(list))
    for folder,label in [('evaluation_best_flow','best_2000'),('evaluation','latest_6000')]:
        rows=[json.loads(line) for line in (ROOT/folder/'metrics.jsonl').open()]
        assert len(rows)==len(cases)*2
        for r in rows:
            arm,group=r['mode'].split(':',1)
            if arm=='before' and label=='latest_6000':continue
            metrics[group]['before' if arm=='before' else label].append(r['metrics'])
    differences=[]
    for i in range(len(cases)):
        a=np.array(Image.open(ROOT/'evaluation/images'/f'before_{i:03d}.png'))
        b=np.array(Image.open(ROOT/'evaluation_best_flow/images'/f'before_{i:03d}.png'))
        differences.append(int(np.abs(a.astype(int)-b.astype(int)).max()))
    summary={}
    for group,arms in metrics.items():
        summary[group]={}
        for arm,rows in arms.items():
            values={}
            for key in rows[0]:
                observed=[r[key] for r in rows if r[key] is not None]
                if observed:values[key]=float(np.mean(observed))
            summary[group][arm]={'n':len(rows),**values}
    report={'groups':summary,'baseline_max_rgba_difference_between_devices':max(differences),
            'cases':len(cases),'scope':'Fixed development probes; pixel distance is not a complete measure of generative quality. Broad/old subsets overlap; no locked test. Training stopped on user deadline, not full 200k schedule.',
            'steps':{'before':8000,'best':2000,'latest':6000},'expected_text_draws_per_image_at_6000':6000*64*.9/324921}
    if (ROOT/'parent_flow.json').exists():
        flow=json.load(open(ROOT/'parent_flow.json'))['metrics']['text_flow_mse']
        logs=[json.loads(line) for line in (ROOT/'training/metrics.jsonl').open()]
        report['same_1024_validation_flow']={'before':flow,**{str(r['step']):r['text_flow_mse'] for r in logs if 'text_flow_mse' in r}}
    atomic_json(ROOT/'comparison.json',report)
    header='''<!doctype html><meta charset="utf-8"><title>全量训练停止后评估</title><style>body{font:15px sans-serif;margin:24px;background:#eee}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:8px}img{width:128px;height:128px;image-rendering:pixelated;background:#999}td:first-child{max-width:420px}th{position:sticky;top:0;background:white}</style><h1>扩容前 / best 2000 / latest 6000</h1><p>同一提示、种子、32 Heun 步、CFG 2。开发集评估，未使用锁定测试集。</p><table><tr><th>组 / 提示</th><th>扩容前 best</th><th>全量 best 2000</th><th>全量 latest 6000</th></tr>'''
    lines=[header]
    for i,c in enumerate(cases):
        cells=[f'<tr><td>{html.escape(c["group"])}<br>{html.escape(c["evaluation_prompt"])}<br>seed {c["seed"]}</td>']
        for folder,arm in [('evaluation','before'),('evaluation_best_flow','recovery'),('evaluation','recovery')]:
            cells.append(f'<td><img loading="lazy" src="{folder}/images/{arm}_{i:03d}.png"></td>')
        lines.append(''.join(cells)+'</tr>')
    (ROOT/'comparison.html').write_text('\n'.join(lines)+'</table>',encoding='utf8')
    for group in ['novel_composition','fullscale_val_filename','broad_name','broad_rich']:
        indices=[i for i,c in enumerate(cases) if c['group']==group]
        # All novel probes and first 24 broad/long-tail probes, fixed manifest order.
        indices=indices if group=='novel_composition' else indices[:24]
        for page in range(0,len(indices),12):
            chunk=indices[page:page+12];sheet=Image.new('RGB',(820,35+len(chunk)*105),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),'Prompt                              Before best       Best 2000         Latest 6000',fill='black')
            for row,i in enumerate(chunk):
                y=35+row*105;c=cases[i]
                draw.multiline_text((8,y),'\n'.join(textwrap.wrap(c['evaluation_prompt'],38)[:6]),fill='black')
                for col,(folder,arm) in enumerate([('evaluation','before'),('evaluation_best_flow','recovery'),('evaluation','recovery')]):
                    im=Image.open(ROOT/folder/'images'/f'{arm}_{i:03d}.png').convert('RGBA').resize((96,96),Image.Resampling.NEAREST)
                    bg=Image.new('RGB',(96,96),'#999999');bg.paste(im,mask=im.getchannel('A'));sheet.paste(bg,(340+col*155,y))
            sheet.save(ROOT/f'comparison_{group}_{page//12}.png')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
