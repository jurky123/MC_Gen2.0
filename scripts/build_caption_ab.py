"""Build paired caption ablations only after the frozen image review has a recorded decision."""
import argparse
import json
from pathlib import Path
from mcgen.catalog import write_catalog,file_hash,atomic_json
from mcgen.semantic_captions import filename_words


def build(root,out):
    review=json.loads((root/'quality_review.json').read_text())
    frozen=json.loads((root/'fixed_review64.json').read_text())['cases']
    if review.get('decision')!='proceed_with_controlled_pilot':raise ValueError('Fixed review has not authorized the research comparison')
    if {r['id'] for r in review['ratings']}!={r['id'] for r in frozen}:raise ValueError('Review does not cover the frozen 64 cases')
    if review['revised_output_sha256']!=file_hash(root/'refined/semantic/candidates.jsonl'):raise ValueError('Annotations changed after review')
    cases=[json.loads(s) for s in (root/'vision/cases.jsonl').read_text().splitlines()]
    labels={r['id']:r for r in map(json.loads,(root/'refined/semantic/candidates.jsonl').read_text().splitlines())}
    exclusions={r['id'] for r in review['ratings'] if r['new_rating'] in ('clear_contradiction','unsupported_detail')}
    rows={'old':[],'new':[]};dropped=[]
    for c in cases:
        a=labels[c['id']]
        if c['target_hash']!=a['target_hash']:raise ValueError('Target hash mismatch')
        bad=bool(set(a['quality_flags']) & {'name_conflict','review_schema_error','background_in_caption'}) or c['id'] in exclusions
        if c['split']=='train' and bad:
            dropped.append({'id':c['id'],'flags':a['quality_flags'],'fixed_review_unreliable':c['id'] in exclusions});continue
        for arm in rows:
            prompt=' '.join(filename_words(c['source_filename'])) if c['split']=='val' else (c['old_caption'] if arm=='old' else a['combined_text'])
            if not prompt.strip():raise ValueError('Empty prompt')
            rows[arm].append({**c,'prompt':prompt,'reference':None,
                'caption_experiment_arm':arm,'caption_supervision':'automatic_weak_label_research_pilot' if arm=='new' and c['split']=='train' else 'baseline_or_common_validation',
                'caption_annotation_flags':a['quality_flags']})
    for arm in rows:
        write_catalog(out/arm,rows[arm],{'native_size':16,'parent_protocol':str(root/'protocol.json'),
            'fixed_review_sha256':file_hash(root/'quality_review.json'),'arm':arm,
            'policy':'Same targets/order in both arms. Training excludes explicit conflicts, missing review outputs, background contamination and fixed-review contradictions/unsupported details. Name uncertainty alone is not an exclusion. Validation uses common filename-only prompts.'})
    assert [(r['id'],r['target_hash']) for r in rows['old']]==[(r['id'],r['target_hash']) for r in rows['new']]
    assert [r['prompt'] for r in rows['old'] if r['split']=='val']==[r['prompt'] for r in rows['new'] if r['split']=='val']
    atomic_json(out/'comparison_protocol.json',{'records_per_arm':len(rows['old']),'dropped_train':dropped,
        'train_count':sum(r['split']=='train' for r in rows['old']), 'val_count':sum(r['split']=='val' for r in rows['old']),
        'fixed_review_sha256':file_hash(root/'quality_review.json'),'same_target_order':True,'same_validation_prompts':True})
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();build(a.root,a.out)
