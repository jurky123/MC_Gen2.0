"""Render the preselected review sample without selecting by model outcome."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from audit_item_captions import array
from mcgen.catalog import file_hash
from mcgen.images import composite


def render(root, available_only=False, unreviewed=False):
    protocol=json.loads((root/'protocol.json').read_text())
    assert file_hash(root/'fixed_review64.json')==protocol['fixed_review_sha256']
    picks=json.loads((root/'fixed_review64.json').read_text())['cases']
    cases={r['id']:r for r in map(json.loads,(root/'vision/cases.jsonl').read_text().splitlines())}
    revised={r['id']:r for r in map(json.loads,(root/'refined/semantic/candidates.jsonl').read_text().splitlines())}
    selected=[(i,p) for i,p in enumerate(picks) if not available_only or p['id'] in revised]
    if unreviewed:
        done={r['id'] for r in json.loads((root/'quality_review_partial.json').read_text())['ratings']}
        selected=[(i,p) for i,p in selected if p['id'] not in done]
    dirname=f'fixed_review_remaining_{len(selected)}' if unreviewed else f'fixed_review_partial_{len(selected)}' if available_only else 'fixed_review'
    out=root/dirname;out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    def text(draw,value,x,y,width):
        line=''
        for char in value:
            if char=='\n' or font.getlength(line+char)>width:
                draw.text((x,y),line,font=font,fill='black');y+=23;line='' if char=='\n' else char
            else:line+=char
        if line:draw.text((x,y),line,font=font,fill='black')
    rows=[]
    for start in range(0,len(selected),8):
        sheet=Image.new('RGB',(1500,8*190),'white');draw=ImageDraw.Draw(sheet)
        for k,(index,p) in enumerate(selected[start:start+8]):
            c=cases[p['id']];r=revised[p['id']];assert c['target_hash']==p['target_hash']==r['target_hash']
            from mcgen.images import composite
            a=array(c);y=k*190
            sheet.paste(Image.fromarray(np.uint8(np.rint(composite(a,128)))).resize((128,128),Image.Resampling.NEAREST),(4,y+30))
            text(draw,f"{index} {p['split']}",4,y,150)
            text(draw,'NAME: '+c['source_filename']+'\nOLD: '+c['old_caption'],155,y,435)
            text(draw,'NEW: '+r['combined_text']+'\nFLAGS: '+', '.join(r['quality_flags']),620,y,870)
            draw.line((0,y+185,1500,y+185),fill='#ddd')
            rows.append(dict(index=index,**p,filename=c['source_filename'],old_caption=c['old_caption'],
                new_caption=r['combined_text'],flags=r['quality_flags']))
        sheet.save(out/f'page_{start//8:02d}.png')
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'Rendered {len(rows)} frozen review cases in {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--available-only',action='store_true')
    p.add_argument('--unreviewed',action='store_true')
    a=p.parse_args();render(a.root,a.available_only,a.unreviewed)
