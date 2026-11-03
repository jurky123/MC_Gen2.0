"""Freeze an unseen, project-isolated expansion and a preselected review sample."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import re
import shutil
import pyarrow.parquet as pq
from mcgen.catalog import read_catalog,audit,atomic_json,file_hash
from mcgen.color_supervision import category_from_caption

ROOT=Path(__file__).resolve().parents[1]

def prepare(out):
    if out.exists():raise ValueError('Use a fresh output directory')
    records,meta=read_catalog(ROOT/'data/catalogs/native16_v3/catalog')
    names=pq.read_table(ROOT/'data/build/mc_text2image32_wl/metadata.parquet',columns=['file_name','mod_slug']).to_pandas()
    views=pq.read_table(ROOT/'data/build/mc_text2image32_wl/prompt_views.parquet',columns=['index','prompt_0','prompt_3']).to_pandas().set_index('index')
    excluded=set()
    for d in ['text_caption_audit_20261102','semantic_caption_review_20261102']:
        excluded.update(json.loads(s)['target_hash'] for s in (ROOT/'outputs'/d/'cases.jsonl').read_text().splitlines())
    unique={}
    for r in records:
        if r['asset_type']=='item' and r['split'] in ('train','val') and r['target_hash'] not in excluded:
            unique.setdefault(r['target_hash'],r)
    rng=random.Random(20261103);selected=[];used=set();projects=Counter()
    for split,total in [('train',1024),('val',128)]:
        pool=[r for r in unique.values() if r['split']==split];rng.shuffle(pool);chosen=[]
        def take(r,method):
            if r['target_hash'] in used or projects[r['lineage']]>=12:return False
            chosen.append((r,method));used.add(r['target_hash']);projects[r['lineage']]+=1;return True
        for r in pool:
            take(r,'uniform_unique_target')
            if len(chosen)==total//2:break
        groups=defaultdict(list)
        for r in pool:
            filename=str(names.iloc[r['source_row']]['file_name'])
            family=category_from_caption(re.sub(r'[_\-]+',' ',Path(filename).stem))
            groups[family].append(r)
        while len(chosen)<total and groups:
            for family in list(sorted(groups)):
                while groups[family]:
                    if take(groups[family].pop(),'filename_stratum:'+family):break
                else:del groups[family]
                if len(chosen)==total:break
        if len(chosen)!=total:raise ValueError('Insufficient sources under project cap')
        for r,method in chosen:
            i=r['source_row'];v=views.loc[i]
            selected.append({**r,'reference':None,'source_filename':str(names.iloc[i]['file_name']),
                'source_mod':str(names.iloc[i]['mod_slug']),'old_caption':r['prompt'],
                'previous_factual':str(v['prompt_0']),'previous_descriptive':str(v['prompt_3']),
                'selection_method':method})
    report=audit(selected)
    out.mkdir(parents=True);vision=out/'vision';vision.mkdir();(vision/'download').mkdir()
    cases=vision/'cases.jsonl';cases.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    fixed=[];review_rng=random.Random(20261104)
    for split,count in [('train',48),('val',16)]:
        subset=[r for r in selected if r['split']==split];review_rng.shuffle(subset)
        fixed.extend({'id':r['id'],'target_hash':r['target_hash'],'split':split} for r in subset[:count])
    review_path=out/'fixed_review64.json';atomic_json(review_path,{'selection':'Frozen before annotation; 48 train + 16 val, uniform within each selected split','cases':fixed})
    protocol={'parent_catalog_fingerprint':meta['fingerprint'],'cases_sha256':file_hash(cases),
        'annotator':'Qwen/Qwen3-VL-8B-Instruct','annotator_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        'selection':'Half uniform unique targets, half filename-family strata; max 12 targets/project. Not a population-weighted sample.',
        'excluded_previously_reviewed_targets':len(excluded),'fixed_review_sha256':file_hash(review_path),
        'audit':report,'test_used':False,'training_plan':'Same target set and initialization; change only text supervision after fixed review. No training before review.',
        'generation_model':'Qwen3-8B unchanged'}
    atomic_json(vision/'protocol.json',protocol);atomic_json(out/'protocol.json',protocol)
    for filename in ['verified.json','manifest.json']:
        shutil.copyfile(ROOT/'outputs/semantic_caption_batch256_20261102/download'/filename,vision/'download'/filename)
    print(json.dumps(protocol,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);prepare(p.parse_args().out)
