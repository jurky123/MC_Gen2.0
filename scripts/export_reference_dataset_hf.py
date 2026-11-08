"""Build portable, split-preserving reference datasets for the existing HF repo."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq

from mcgen.catalog import atomic_json, file_hash
from mcgen.images import content_hash, rgba

RELEASE='reference-20261107'
SOURCE=Path('outputs/reference_mixed_hq_20261107')
ACTIVE=Path('outputs/reference_native16_lossless_20261107')
BUNDLE=Path('data/raw/MC_Gen_AllData_20261106')
ARCHIVES=('high_quality_native16.zip','reviewed_legacy_native16.zip','high_quality_native32.zip')


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows))


def no_local_paths(value):
    if isinstance(value,dict):
        for item in value.values():no_local_paths(item)
    elif isinstance(value,list):
        for item in value:no_local_paths(item)
    elif isinstance(value,str) and (value.startswith(('/home/','/tmp/','/mnt/')) or
                                   (len(value)>2 and value[1]==':' and value[2] in '/\\')):
        raise ValueError('Local filesystem path in a public record')


def build(root):
    root.mkdir(parents=True,exist_ok=True)
    destination=root/'hf'/'data'/RELEASE
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('archives','manifests','metadata'):(destination/name).mkdir(exist_ok=True)
    base=json.loads((SOURCE/'pairs.json').read_text())
    active=json.loads((ACTIVE/'pairs.json').read_text())
    active_by_id={r['id']:r for r in active}
    raw={r['sample_id']:r for r in map(json.loads,(BUNDLE/'metadata/pairs_main_1000.jsonl').read_text().splitlines())}
    legacy=json.loads(Path('outputs/spatial_reference_20261104/pairs.json').read_text())
    source_audit=json.loads((SOURCE/'data_audit.json').read_text())
    active_audit=json.loads((ACTIVE/'data_audit.json').read_text())
    assert file_hash(SOURCE/'pairs.json')==source_audit['pairs_sha256']
    assert file_hash(ACTIVE/'pairs.json')==active_audit['pairs_sha256']
    contract={'release':RELEASE,'source_pairs_sha256':source_audit['pairs_sha256'],
              'native16_pairs_sha256':active_audit['pairs_sha256'],'export_protocol':'portable_reference_v1'}
    contract_path=root/'export_source.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise ValueError('Export directory belongs to another source version')
    atomic_json(contract_path,contract)
    columns=['index','file_name','type','project_id','mod_slug','author','license','project_type','version_url','weak_prompt']
    table=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=columns)
    indices=sorted({r['source_row'] for r in base if r.get('source_row') is not None})
    origins={r['index']:r for r in table.take(indices).to_pylist()}
    assert set(origins)==set(indices)
    source_images=np.memmap('data/build/mc_text2image32_wl/images.uint8.mmap',mode='r',dtype='uint8',shape=(len(table),32,32,4))
    del table
    assets={}
    members=defaultdict(set)
    def register(path,relative,archive):
        path=Path(path)
        if relative not in assets:
            with Image.open(path) as im:
                size=list(im.size);mode=im.mode
            assets[relative]={'file':relative,'sha256':file_hash(path),'bytes':path.stat().st_size,
                              'size':size,'mode':mode,'source':path,'archives':[]}
        elif assets[relative]['source'].resolve()!=path.resolve() and file_hash(path)!=assets[relative]['sha256']:
            raise ValueError('Conflicting files at shared asset path')
        members[archive].add(relative)
        if archive not in assets[relative]['archives']:assets[relative]['archives'].append(archive)
        return relative
    all32=[]
    filename_differences=[]
    checked_source_rows=set()
    original_to_public={}
    for index,row in enumerate(base):
        hq=row['quality_tier']=='highest_quality'
        retained=row['id'] in active_by_id
        archive=ARCHIVES[0] if hq and retained else ARCHIVES[2] if hq else ARCHIVES[1]
        target32=register(row['target_file'],f'assets/targets32/{row["target_hash"]}.png',archive)
        with Image.open(row['target_file']) as im:large=rgba(im)
        assert content_hash(large)==row['target_hash'] and large.shape==(32,32,4)
        if row.get('source_row') is not None and row['source_row'] not in checked_source_rows:
            assert np.array_equal(rgba(source_images[row['source_row']]),large)
            checked_source_rows.add(row['source_row'])
        reference=register(row['reference_file'],f'assets/references/{row["reference_sha256"]}.png',archive)
        assert assets[reference]['sha256']==row['reference_sha256']
        record={k:copy.deepcopy(row[k]) for k in ('id','quality_tier','quality_basis','source_family',
                   'split','split_group','source_row','captions') if k in row}
        record.update(asset_type='item',target_file=target32,target_hash=row['target_hash'],
            target_file32=target32,target_hash32=row['target_hash'],target_size=[32,32],
            target_png_sha256=assets[target32]['sha256'],reference_file=reference,
            reference_sha256=assets[reference]['sha256'],reference_size=assets[reference]['size'],
            caption_short=row['captions'][0],caption_detailed=row['captions'][-1],
            included_in_native16=retained,source_pair_index32=index)
        if retained:
            r=active_by_id[row['id']]
            assert r['target_conversion']['lossless'] and r['reference_sha256']==row['reference_sha256']
            assert r['captions']==row['captions'] and r['split']==row['split']
            small_file=register(r['target_file'],f'assets/targets16/{r["target_hash"]}.png',archive)
            with Image.open(r['target_file']) as im:small=rgba(im)
            assert small.shape==(16,16,4) and np.array_equal(small.repeat(2,0).repeat(2,1),large)
            record.update(target_file16=small_file,target_hash16=r['target_hash'],
                          target_png_sha256_16=assets[small_file]['sha256'],native16_conversion='exact_repeat_collapse')
        else:
            record.update(target_file16=None,target_hash16=None,native16_conversion=None,
                          native16_exclusion_reason='genuine32_detail_requires_lossy_conversion')
        origin=copy.deepcopy(origins.get(row.get('source_row'),row['provenance']))
        provenance={'original_source':origin,'upstream_dataset':'Risposta/MC_Gen',
                    'upstream_revision':'e1b7f0499c6fd1b9b35e64ca7cdcc8bed65b4d99'}
        if hq:
            original=raw[row['id']]
            original_path=BUNDLE/original['mc_image']
            original_sha=file_hash(original_path)
            original_file=register(original_path,f'assets/original_mc/{original_sha}.png',archive)
            with Image.open(original_path) as im:assert np.array_equal(rgba(im),large)
            record.update(original_mc_file=original_file,original_mc_sha256=original_sha)
            for key in ('name_en','name_zh','hd_annotation_en','hd_annotation_zh','mc_annotation_en',
                        'mc_annotation_zh','mc_annotation_basis','expanded_generation_prompt','quality_status'):
                record[key]=original.get(key)
            provenance['delivery_provenance']=copy.deepcopy(row['provenance'])
            provenance['delivery_sample_id']=original['sample_id']
            record['caption_basis']=original['mc_annotation_basis']
            record['reference_generation']={'source':'user_curated_delivery','provided_filename':original['hd_image']}
        else:
            old=legacy[row['legacy_feature_index']]
            assert old['reference_sha256']==row['reference_sha256']
            record['caption_basis']='existing_filename_resource_supervision'
            provenance.update(label_source_filename=old['source_filename'],original_source_filename=origin['file_name'],
                              review_revision=1182,review_policy='human_keep' if row['quality_tier']=='reviewed' else 'evaluation_only')
            record['reference_generation']={k:old.get(k) for k in ('reference_generator','generator','generator_license')}
            if old['source_filename']!=origin['file_name']:
                note={'id':row['id'],'source_row':row['source_row'],'label_source_filename':old['source_filename'],
                      'anchored_original_filename':origin['file_name'],'target_pixels_match_original_source':True}
                filename_differences.append(note)
                provenance['filename_note']='Label-source filename differs from the source-row anchor; both retained without relabeling.'
        record['provenance']=provenance
        no_local_paths(record)
        all32.append(record)
        original_to_public[row['id']]=record
        if (index+1)%500==0:print('PREPARED PAIRS',index+1,len(base),flush=True)
    native16=[]
    for r in active:
        record=copy.deepcopy(original_to_public[r['id']])
        record.update(target_file=record['target_file16'],target_hash=record['target_hash16'],target_size=[16,16],
                      target_png_sha256=record['target_png_sha256_16'],
                      target_conversion={'method':'exact_repeat_collapse','lossless':True,
                                         'source_canvas':[32,32],'output_canvas':[16,16]})
        native16.append(record)
    assert len(all32)==2970 and len(native16)==2470
    owners={}
    for row in native16:
        for kind,key in [('target',row['target_hash']),('reference',row['reference_sha256']),('source_row',row.get('source_row'))]:
            if key is not None:assert owners.setdefault((kind,key),row['split'])==row['split']
    sets={'all_sources32':all32,'high_quality_originals':[r for r in all32 if r['quality_tier']=='highest_quality'],
          'reviewed_legacy':[r for r in native16 if r['source_family']=='reviewed_legacy'],
          'native16_all':native16,'excluded_true32':[r for r in all32 if not r['included_in_native16']]}
    for split in ('train','val','test'):sets['native16_'+split]=[r for r in native16 if r['split']==split]
    sets['native16_text_train']=[r for r in native16 if r['split']=='train' and r['quality_tier']=='highest_quality']
    for name,rows in sets.items():jsonl(destination/'manifests'/f'{name}.jsonl',rows)
    jsonl(destination/'metadata/filename_source_notes.jsonl',filename_differences)
    image_index=[{k:v for k,v in asset.items() if k!='source'} for _,asset in sorted(assets.items())]
    jsonl(destination/'metadata/images.jsonl',image_index)
    snapshot=json.loads((ACTIVE/'review_snapshot.json').read_text())
    atomic_json(destination/'metadata/review_selection.json',{
        k:snapshot[k] for k in ('revision','source_pairs_sha256','selection_policy','selected_pair_ids')})
    (destination/'metadata/source_delivery_readme.md').write_text(
        '# 原始交付说明（存档）\n\n本发布仅提取正式主数据1000对，不包含下文所述历史、淘汰图或模糊输入。\n\n'+
        (BUNDLE/'README_汇总说明.md').read_text())
    archive_info=[]
    for name in ARCHIVES:
        path=destination/'archives'/name
        tmp=path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for relative in sorted(members[name]):z.write(assets[relative]['source'],relative)
        tmp.replace(path)
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist())==members[name]
            for member in z.infolist():
                digest=hashlib.sha256()
                with z.open(member) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
                assert digest.hexdigest()==assets[member.filename]['sha256']
        archive_info.append({'file':'archives/'+name,'bytes':path.stat().st_size,'sha256':file_hash(path),'members':len(members[name])})
        print('ARCHIVE VERIFIED',name,path.stat().st_size,flush=True)
    summary={**contract,'format':'relative-path JSONL + original PNG bytes in ZIP archives',
        'primary_high_quality_pairs':1000,'high_quality_native16_pairs':500,'high_quality_true32_pairs':500,
        'reviewed_train_pairs':692,'legacy_evaluation_only_pairs':1278,
        'native16_splits':dict(Counter(r['split'] for r in native16)),
        'native16_unique_targets':{s:len({r['target_hash'] for r in native16 if r['split']==s}) for s in ('train','val','test')},
        'native16_text_train_pairs':len(sets['native16_text_train']),'lossy_native16_pairs':0,
        'preserved_hq_original_mc_pngs':1000,'source_image_rows_verified':len(checked_source_rows),
        'filename_source_notes':len(filename_differences),'assets':len(assets),'archives':archive_info,
        'manifests':{name:{'file':f'manifests/{name}.jsonl','rows':len(rows)} for name,rows in sets.items()},
        'caption_policy':'MC captions and short names only for text supervision; HD captions and generation prompts stored separately',
        'quality_policy':'highest_quality denotes user-curated primary pairs; legacy validation remains evaluation_only',
        'license':'Research/study release consistent with existing repository; original assets retain their per-source licenses and attribution',
        'source_splits_and_training_captions_preserved':True,'feature_caches_or_model_weights_included':False}
    no_local_paths(summary)
    atomic_json(destination/'dataset.json',summary)
    render_preview(destination,native16,assets)
    (destination/'README.md').write_text(readme())
    files=sorted(p for p in destination.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (destination/'SHA256SUMS.txt').write_text(''.join(f'{file_hash(p)}  {p.relative_to(destination).as_posix()}\n' for p in files))
    atomic_json(root/'export_verification.json',{'passed':True,'source_files_changed':False,
        'relative_paths_only':True,'archive_member_sha256_checks':sum(a['members'] for a in archive_info),
        'source_image_pixels_verified':len(checked_source_rows),'lossless16_roundtrips':len(native16),
        'cross_split_collisions':0,'all_native16_targets_exact':True,'summary':summary})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def render_preview(destination,rows,assets):
    import random
    rng=random.Random(202611074)
    selected=[]
    for quality,split,count in [('highest_quality','train',3),('reviewed','train',2),
                                 ('highest_quality','val',2),('highest_quality','test',1)]:
        unique={r['target_hash']:r for r in rows if r['quality_tier']==quality and r['split']==split}
        selected+=rng.sample(list(unique.values()),count)
    canvas=Image.new('RGB',(1040,68+152*len(selected)),'#111827')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    draw.text((14,12),'Reference training data | HD reference / exact native16 target / actual training captions',font=font,fill='white')
    for j,row in enumerate(selected):
        y=52+j*152
        for c,key in enumerate(('reference_file','target_file')):
            with Image.open(assets[row[key]]['source']) as im:
                im=im.convert('RGBA');bg=Image.new('RGBA',im.size,'#edf0f4');bg.alpha_composite(im)
                pic=bg.convert('RGB').resize((112,112),Image.Resampling.LANCZOS if c==0 else Image.Resampling.NEAREST)
                canvas.paste(pic,(14+c*128,y))
        draw.text((284,y),row['quality_tier']+' / '+row['split'],font=font,fill='#93c5fd')
        draw.text((284,y+20),'Short: '+row['caption_short'],font=font,fill='white')
        lines=textwrap.wrap('Detail: '+row['caption_detailed'],width=95)
        for k,line in enumerate(lines[:6]):draw.text((284,y+40+k*15),line,font=font,fill='#cbd5e1')
    canvas.save(destination/'preview.png')


def readme():
    return '''# MCGen 参考图训练数据 · 2026-11-07

高清参考、MC目标、文本和来源一起保存，可用于参考图生成与文生图。质量等级与原训练/验证/测试划分保持不变。

![实际数据抽样](preview.png)

| 下载包 | 内容 | 使用方式 |
|---|---|---|
| `high_quality_native16.zip` | 用户精筛的500对装备，原MC为精确两倍放大，已无损恢复16×16 | 当前16格训练 |
| `reviewed_legacy_native16.zip` | 人工确认的692对旧训练配对，以及仅评估的1278对旧验证配对 | 当前16格训练/验证 |
| `high_quality_native32.zip` | 另500对真实32格高清配对，保留原尺寸 | 独立原件，不进入16格训练 |

下载包均在 `archives/`。当前16格版本只需前两包。解压到本目录，自动合并为 `assets/`；少量重复目标路径的文件内容相同。

```python
from pathlib import Path
from zipfile import ZipFile

root = Path("data/reference-20261107")
for name in ("high_quality_native16.zip", "reviewed_legacy_native16.zip"):
    with ZipFile(root / "archives" / name) as archive:
        archive.extractall(root)
```

| 16格清单 | 配对数 | 独立目标 |
|---|---:|---:|
| `manifests/native16_train.jsonl` | 1064 | 864 |
| `manifests/native16_val.jsonl` | 1350 | 711 |
| `manifests/native16_test.jsonl` | 56 | 56 |

`native16_text_train.jsonl`包含372对最高质量训练数据，供混入文生图任务。`high_quality_originals.jsonl`保留完整1000对主数据；`excluded_true32.jsonl`明确列出不进入16格版本的500对。未收录历史/淘汰素材、未经确认的旧训练配对或任何32→16有损候选。

清单中的`target_file`与`reference_file`均相对本目录。`captions`、`caption_short`和`caption_detailed`是实际训练标签；MC观察、配对语义复核、HD观察与生成意图分列，不把高清描述冒充MC监督。旧数据没有额外精标时，长短文本相同。

高清参考与新批次`original_mc_file`保留原PNG字节；训练目标使用无损16格，`target_file32`另存原训练32格缓存。原始文件名、标签来源文件名、模组、作者、来源链接和已有许可信息保留在`provenance`。部分去重条目的标签文件名与来源行锚点不同，二者分列；`metadata/filename_source_notes.jsonl`记录差异，其目标均已逐像素核对来源。

`metadata/images.jsonl`和`SHA256SUMS.txt`提供校验。图片不包含模型、优化器或特征缓存。沿用仓库的研究/学习用途声明；来源素材保留各自许可与署名信息，本汇总不改变原素材授权。
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/hf_reference_release_20261107'))
    build(parser.parse_args().out)
