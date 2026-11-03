"""Render an evidence-separated, assistant-reviewed caption pilot; never changes training data."""
import argparse
import base64
from collections import Counter
import csv
import html
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mcgen.catalog import file_hash
from mcgen.images import native_grid, content_hash, composite

ROOT = Path(__file__).resolve().parents[1]
STATUS = {'candidate':'候选待确认','identity_review':'身份待核实','name_review':'名称待核实','style_review':'风格待核实'}


def build(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if file_hash(out/'cases.jsonl')!=protocol['case_sha256']:
        raise ValueError('case manifest changed')
    cases=[json.loads(x) for x in (out/'cases.jsonl').read_text().splitlines()]
    notes=json.loads((out/'review_notes.json').read_text())
    old=[json.loads(x) for x in (ROOT/'outputs/text_caption_audit_20261102/assistant_review.jsonl').read_text().splitlines()]
    blind=json.loads((out/'blind_observations.json').read_text())
    assert len(cases)==len(notes)==32
    maps={};records=[];pictures=[]
    for i,(r,n) in enumerate(zip(cases,notes)):
        name,zh,visual_zh,status,note=n
        assert status in STATUS
        if i<24:
            assert old[i]['target_hash']==r['target_hash']
            visual=old[i]['corrected_short']
            method='Prior assistant visual review with names visible; visual text reused.'
        else:
            assert blind[i-24]['index']==i
            visual=blind[i-24]['visual_en']
            method='Assistant visual observation recorded before opening this case filename; theme selection was known.'
        spec=r['target'];key=(spec['path'],tuple(spec['shape']))
        if key not in maps:maps[key]=np.memmap(key[0],dtype=np.uint8,mode='r',shape=key[1])
        a=native_grid(np.array(maps[key][spec['index']]),16)
        assert content_hash(a)==r['target_hash']
        picture=Image.new('RGB',(256,128))
        for j,bg in enumerate((245,32)):
            picture.paste(Image.fromarray(np.uint8(np.rint(composite(a,bg)))).resize((128,128),Image.Resampling.NEAREST),(128*j,0))
        pictures.append(picture)
        baseline=re.sub(r'[_.\-]+',' ',Path(r['source_filename']).stem).strip()
        combined=f'{name}. {visual}'
        records.append(dict(index=i+1,id=r['id'],target_hash=r['target_hash'],split=r['split'],
            source_filename=r['source_filename'],filename_only_baseline=baseline,
            existing_training_caption=r['old_caption'],semantic_name_en=name,semantic_name_zh=zh,
            name_evidence={'source':'metadata.file_name','source_row':r['source_row'],'mod':r['source_mod'],
                           'official_localization_verified':False,'raw':r['source_filename']},
            visual_en=visual,visual_zh=visual_zh,combined_en=combined,
            combined_zh=f'{zh}。{visual_zh}',status=status,review_note_zh=note,
            proposed_training_views=[name,visual,combined] if status=='candidate' else [],
            reviewer='assistant_visual_review',review_method=method,
            approved_for_training=False,user_decision='pending'))
    (out/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    columns=['index','id','source_filename','filename_only_baseline','existing_training_caption',
             'semantic_name_en','semantic_name_zh','visual_en','visual_zh','combined_en','combined_zh',
             'status','review_note_zh','target_hash','user_decision']
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    csv_text=buffer.getvalue();(out/'comparison.csv').write_text(csv_text,encoding='utf-8-sig')
    cards=[]
    for r,pic in zip(records,pictures):
        b=io.BytesIO();pic.save(b,format='PNG');uri='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        e=html.escape
        cards.append(f'''<article class="card {r['status']}" data-status="{r['status']}">
<header><b>#{r['index']:02d} {e(r['source_filename'])}</b><span>{STATUS[r['status']]}</span></header>
<div class="grid"><div><img src="{uri}" alt="{e(r['source_filename'])}"><small>同一图片，浅色／深色背景</small></div>
<div><h3>文件名直接提取</h3><p>{e(r['filename_only_baseline'])}</p><h3>当前旧训练描述</h3><p>{e(r['existing_training_caption'])}</p></div>
<div><h3>保留的名称语义</h3><p>{e(r['semantic_name_en'])}</p><p>{e(r['semantic_name_zh'])}</p><h3>独立视觉描述</h3><p>{e(r['visual_en'])}</p><p>{e(r['visual_zh'])}</p></div>
<div><h3>合并描述候选</h3><p>{e(r['combined_en'])}</p><p>{e(r['combined_zh'])}</p></div></div>
<footer>{e(r['review_note_zh'])}</footer></article>''')
    csv_uri='data:text/csv;charset=utf-8;base64,'+base64.b64encode(('\ufeff'+csv_text).encode()).decode()
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC 物品文本监督对照 · 32 张</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#182530;background:#f1f4f6;margin:24px auto;max-width:1600px;padding:0 16px}h1{font-size:28px}h3{font-size:14px;color:#526679;margin:0}p{margin:6px 0 14px}small{display:block;color:#667}button,a{margin-right:12px}button{padding:8px 16px;cursor:pointer}header{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #ddd;padding-bottom:12px}header b{overflow-wrap:anywhere}header span{white-space:nowrap}.card{background:white;border:1px solid #ccd5dc;border-left:5px solid #299379;border-radius:8px;margin:18px 0;padding:18px}.identity_review,.name_review,.style_review{border-left-color:#cd8b22}.grid{display:grid;grid-template-columns:256px 1fr 1.2fr 1.2fr;gap:22px;padding:20px 0}.grid p{overflow-wrap:anywhere}img{width:256px;image-rendering:pixelated}footer{background:#f3f5f7;padding:10px}.intro{max-width:1100px}@media(max-width:1000px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>
<h1>MC 物品文本监督对照 · 32 张</h1><div class="intro"><p>核对重点：是否保留了有价值的名称概念？新增外观是否真的可见？名称和外观冲突是否被明确指出？</p><p>本批由助手逐图标注，未运行本地视觉标注模型。名称证据来自文件名，未验证官方语言文件；中文为核对释义。前 24 张复用已看过名称的视觉审查，后 8 张先记录视觉观察再查看文件名。全部待确认，未写入训练集；待核实条目的合并文本仅供比较。</p></div>
<button onclick="filter(false)">全部 32 张</button><button onclick="filter(true)">仅看待核实</button>'''
    page+=f'<a download="comparison.csv" href="{csv_uri}">下载完整对照 CSV</a>'+''.join(cards)
    page+='''<script>function filter(flag){document.querySelectorAll('.card').forEach(c=>c.hidden=flag&&c.dataset.status==='candidate')}</script></html>'''
    (out/'comparison.html').write_text(page)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    def wrapped(draw,text,xy,width,font):
        x,y=xy;line=''
        for char in text:
            if char=='\n' or font.getlength(line+char)>width:
                draw.text((x,y),line,font=font,fill='#182530');y+=28;line='' if char=='\n' else char
            else:line+=char
        if line:draw.text((x,y),line,font=font,fill='#182530');y+=28
        return y
    for start in range(0,32,4):
        sheet=Image.new('RGB',(1700,4*300+65),'white');draw=ImageDraw.Draw(sheet)
        draw.text((20,12),'图片  |  文件名直接提取与当前旧描述  |  名称语义 + 视觉描述候选  |  核对说明',font=font,fill='black')
        for k,(r,pic) in enumerate(zip(records[start:start+4],pictures[start:start+4])):
            y=65+k*300;sheet.paste(pic,(12,y+40));draw.text((12,y),f"#{r['index']:02d} {STATUS[r['status']]}",font=font,fill='black')
            wrapped(draw,'文件名：'+r['source_filename']+'\n直接提取：'+r['filename_only_baseline']+'\n当前描述：'+r['existing_training_caption'],(285,y),455,small)
            wrapped(draw,r['combined_en']+'\n'+r['combined_zh'],(770,y),520,font)
            wrapped(draw,r['review_note_zh'],(1320,y),365,font)
            draw.line((0,y+290,1700,y+290),fill='#ccc')
        sheet.save(out/f'comparison_{start//4+1:02d}.png')
    summary={'count':len(records),'status_counts':dict(Counter(r['status'] for r in records)),
             'approved_for_training':0,'local_vlm_inference':False,'test_split_used':False,
             'annotations_sha256':file_hash(out/'annotations.jsonl'),'cases_sha256':file_hash(out/'cases.jsonl')}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs/semantic_caption_review_20261102')
    build(parser.parse_args().out)
