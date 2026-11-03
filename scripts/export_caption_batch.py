"""Export filename / old prompt / independent visual / semantic fusion comparisons."""
import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
import re
from audit_item_captions import array
from mcgen.images import composite
import numpy as np
from PIL import Image

FLAG_LABELS={
    'name_information_loss':'模型删掉了名称信息；已退回保留原词的名称',
    'caption_name_loss':'改写遗漏名称信息；已使用保留原词的描述',
    'name_uncertain':'名称与图像的对应关系仍不确定',
    'name_conflict':'名称与图像可能冲突',
    'background_in_caption':'描述仍含背景信息',
    'review_schema_error':'复核输出格式错误；当前保留原始观察与名称',
    'uncertain_identity':'独立视觉观察未能确定身份',
    'missing_visual_evidence':'缺少有效视觉描述',
    'schema_error':'输出结构不合规',
}


def export(out, previous=None):
    def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
    cases=read(out/'cases.jsonl');visual={r['id']:r for r in read(out/'candidates.jsonl')}
    semantic={r['id']:r for r in read(out/'semantic/candidates.jsonl')}
    prior={r['id']:r for r in read(previous/'semantic/candidates.jsonl')} if previous else {}
    spot_path=out/'assistant_spot_review.json'
    spot=json.loads(spot_path.read_text()) if spot_path.exists() else {}
    notes={r['id']:r.get('note', '') for r in spot.get('findings',[])}
    rows=[];cards=[];e=html.escape
    for case in cases:
        v=visual[case['id']];s=semantic[case['id']];ann=v.get('annotation') or {};name=s.get('semantic') or {}
        a=array(case);pic=Image.new('RGB',(256,128))
        for j,bg in enumerate((245,32)):
            pic.paste(Image.fromarray(np.uint8(np.rint(composite(a,bg)))).resize((128,128),Image.Resampling.NEAREST),(j*128,0))
        b=io.BytesIO();pic.save(b,format='PNG');uri='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        r={'id':case['id'],'filename':case['source_filename'],
           'filename_text':re.sub(r'[_.\-]+',' ',Path(case['source_filename']).stem),
           'old_training_text':case['old_caption'],'visual_short':ann.get('short_caption',''),
           'visual_detailed':ann.get('detailed_caption',''),'proposed_name':name.get('semantic_name',''),
           'retained_name':s.get('effective_name',''),'combined':s.get('combined_text',''),
           'rechecked_visual':s.get('visual_text',''),
           'previous_auto_caption':prior.get(case['id'],{}).get('combined_text',''),
           'format_warnings':', '.join(v.get('format_warnings',[])),
           'failure_reason':s.get('error','') or v.get('error',''),
           'review_note':notes.get(case['id'],''),
           'visual_flags':', '.join(v.get('quality_flags',[])),
           'semantic_flags':', '.join(s.get('quality_flags',[])),
           'status':s['status'],'target_hash':case['target_hash'],'review_decision':'pending'}
        rows.append(r)
        missing='未生成：'+(r['failure_reason'] or '上一步没有有效输出；请查看原始记录')
        cells=[('文件名直接提取',r['filename_text']),('旧训练描述',r['old_training_text']),
               ('独立视觉观察（保留原文）',(r['visual_short']+'\n'+r['visual_detailed']).strip() or missing)]
        if previous:
            cells.append(('上版自动描述',r['previous_auto_caption'] or '上版未生成（原描述曾被格式限制丢弃）'))
        flags='；'.join(FLAG_LABELS.get(f,f) for f in s.get('quality_flags',[]))
        cells.extend([('核对后的外观描述',r['rechecked_visual'] or missing),
               ('最终描述候选',r['combined'] or missing),('待核对',flags or '无自动标记；仍需图像复核')])
        if r['review_note']:cells.append(('逐图复核发现',r['review_note']))
        label='需核对' if r['status']=='quarantine' else '候选待抽查'
        cards.append('<article data-status="'+e(r['status'])+'"><h2>'+e(r['filename'])+' · '+label+'</h2><img src="'+uri+'" alt="'+e(r['filename'])+'"><div class="grid">'+''.join('<section><h3>'+e(title)+'</h3><p>'+e(value)+'</p></section>' for title,value in cells)+'</div></article>')
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>256 张自动标注对照</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:24px auto;padding:0 20px;background:#f4f6f8;color:#1b2934}article{background:white;padding:20px;margin:20px 0;border:1px solid #ccd5dc}h2{font-size:18px}h3{font-size:15px;color:#456}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}p{white-space:pre-wrap;overflow-wrap:anywhere}img{image-rendering:pixelated}button{padding:8px 14px;margin-right:10px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>256 张自动标注对照</h1><p>名称依据为文件名；视觉描述由本地视觉模型独立生成，随后进行名称合并。全部为待审查候选，模型自报相符不代表已确认正确。缺失名称词时保留原词并标记。原训练数据未覆盖。</p><button onclick="show(false)">全部</button><button onclick="show(true)">仅自动标记疑点</button>'''
    page+=''.join(cards)+'''<script>function show(flag){document.querySelectorAll('article').forEach(c=>c.hidden=flag&&c.dataset.status!=='quarantine')}</script></html>'''
    if previous:
        page=page.replace('随后进行名称合并','随后结合文件名重新看图核对；原始视觉观察单独保留')
    page=page.replace('256 张',f'{len(cases)} 张')
    (out/'comparison.html').write_text(page)
    print(f'Exported {len(rows)} records to {out}/comparison.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--previous',type=Path)
    args=p.parse_args();export(args.out,args.previous)
