"""Validate a complete caption revision and render known regression examples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_item_captions import array
from mcgen.catalog import file_hash,atomic_json
from mcgen.images import composite


def summarize(source,out):
    def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
    cases=read(out/'cases.jsonl');old={r['id']:r for r in read(source/'semantic/candidates.jsonl')}
    old_v={r['id']:r for r in read(source/'candidates.jsonl')}
    revised=read(out/'semantic/candidates.jsonl');new={r['id']:r for r in revised}
    recovered={r['id']:r for r in read(out/'candidates.jsonl')}
    note_path=out/'assistant_spot_review.json'
    notes={r['id']:r['note'] for r in json.loads(note_path.read_text()).get('findings',[])} if note_path.exists() else {}
    ids={r['id'] for r in cases}
    assert len(revised)==len(cases)==len(ids)==256
    assert set(new)==set(old)==set(recovered)==ids
    for r in cases:
        assert new[r['id']]['target_hash']==r['target_hash']
        assert recovered[r['id']]['raw']==old_v[r['id']]['raw']
        assert new[r['id']]['combined_text'].strip() and '_' not in new[r['id']]['combined_text']
        assert not new[r['id']]['approved_for_training']
    summary={'records':len(cases),'old_empty_captions':sum(not r.get('combined_text','').strip() for r in old.values()),
        'new_empty_captions':sum(not r.get('combined_text','').strip() for r in revised),
        'old_captions_with_underscores':sum('_' in r.get('combined_text','') for r in old.values()),
        'new_captions_with_underscores':sum('_' in r.get('combined_text','') for r in revised),
        'recovered_observations':sum(old_v[i]['annotation'] is None and recovered[i]['annotation'] is not None for i in ids),
        'new_schema_fallbacks':sum('review_schema_error' in r['quality_flags'] for r in revised),
        'statuses':dict(Counter(r['status'] for r in revised)),
        'flags':dict(Counter(f for r in revised for f in r['quality_flags'])),
        'approved_for_training':0,'quality_accuracy':None,
        'note':'Counts measure text completeness and automatic flags, not semantic accuracy. Known-failure examples are purposive.',
        'previous_sha256':file_hash(source/'semantic/candidates.jsonl'),
        'revised_sha256':file_hash(out/'semantic/candidates.jsonl')}
    atomic_json(out/'validation_summary.json',summary)
    indices=[0,1,2,7,22,250,252,255]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    def text(draw,value,x,y,width):
        line=''
        for char in value:
            if char=='\n' or font.getlength(line+char)>width:
                draw.text((x,y),line,font=font,fill='#172b39');y+=28;line='' if char=='\n' else char
            else:line+=char
        if line:draw.text((x,y),line,font=font,fill='#172b39')
    for page in range(2):
        sheet=Image.new('RGB',(1600,4*260+55),'white');draw=ImageDraw.Draw(sheet)
        draw.text((16,10),'图片与文件名  |  上版自动描述  |  修订后的描述（仍需复核）',font=font,fill='black')
        for k,index in enumerate(indices[page*4:page*4+4]):
            r=cases[index];y=55+k*260;a=array(r)
            sheet.paste(Image.fromarray(np.uint8(np.rint(composite(a,128)))).resize((128,128),Image.Resampling.NEAREST),(15,y+60))
            text(draw,r['source_filename'],15,y,340)
            text(draw,old[r['id']].get('combined_text') or '上版为空：有效观察被格式限制丢弃',375,y+10,540)
            text(draw,new[r['id']]['combined_text'],960,y+10,620)
            if r['id'] in notes:text(draw,'复核：'+notes[r['id']],960,y+115,620)
            draw.line((0,y+250,1600,y+250),fill='#ccc')
        sheet.save(out/f'revision_examples_{page+1}.png')
    with zipfile.ZipFile(out/'caption_revision_256.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['comparison.html','comparison.csv','validation_summary.json','cases.jsonl','candidates.jsonl',
                     'semantic/candidates.jsonl','protocol.json','revision_examples_1.png','revision_examples_2.png']:
            z.write(out/name,name)
        if note_path.exists():z.write(note_path,note_path.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();summarize(args.source,args.out)
