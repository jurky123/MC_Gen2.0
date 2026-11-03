"""Freeze two-caption expansion data and audit actual target-uniform sampling."""
import json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch,yaml
from mcgen.catalog import atomic_json,file_hash,write_catalog
from mcgen.config import Experiment
from mcgen.dataset import CatalogDataset
from mcgen.tasks import TaskSampler
from mcgen.caption_supervision import caption_supervision
from audit_item_captions import array
ROOT=Path('outputs/text_expansion2k_20261102')
CORE=Path('outputs/text_recovery_core_20261102')

def main():
 inputs=[json.loads(s) for s in (ROOT/'annotation_cases.jsonl').read_text().splitlines()]
 labels=[json.loads(s) for s in (ROOT/'annotations.jsonl').read_text().splitlines()]
 byid={r['id']:r for r in labels}
 assert len(byid)==len(labels)==len(inputs), 'Annotations must be complete and unique'
 assert set(byid)=={r['id'] for r in inputs}
 signature=json.loads((ROOT/'annotation_protocol.json').read_text())['signature']
 caption_review=json.loads((ROOT/'caption_quality_review.json').read_text())
 assert caption_review['annotation_sha256']==file_hash(ROOT/'annotations.jsonl')
 expected_indices=set(json.loads((ROOT/'fixed_review64.json').read_text())['indices'])
 assert {r['index'] for r in caption_review['ratings']}==expected_indices
 caption_exclusions={r['id']:r['reason'] for r in caption_review['ratings'] if r['rating'] in ('unsupported_detail','clear_contradiction')}
 review=json.loads((ROOT/'full_visual_review.json').read_text());visual_exclusions={r['id']:r['reason'] for r in review['excluded']}
 selected=[];excluded=[]
 for r in inputs:
  if r['id'] in visual_exclusions:
   excluded.append({'id':r['id'],'reason':visual_exclusions[r['id']]});continue
  a=byid[r['id']];assert a['target_hash']==r['target_hash'] and a['signature']==signature
  policy=caption_supervision(r['source_filename'],a.get('combined_text'),a['quality_flags'],caption_exclusions.get(r['id']))
  selected.append({**r,**policy,'short':policy['filename_text'],'long':policy['descriptive_text'],'raw_descriptive_text':a.get('combined_text'),'caption_quality_flags':a['quality_flags'],'caption_origin':'expansion_automatic_weak_label'})
 core=json.loads((CORE/'cases.json').read_text())
 for r in core:
  policy=caption_supervision(r['source_filename'],r['long'])
  r.update(policy);r['short']=policy['filename_text'];r['raw_descriptive_text']=r['long'];r['long']=policy['descriptive_text'];r['caption_origin']='retained_recovery_core_weak_label'
 assert not {r['target_hash'] for r in core}&{r['target_hash'] for r in selected}
 new_val=[r for r in selected if r['split']=='val'];rows=core+selected
 if (ROOT/'catalog').exists() or (ROOT/'targets.mmap').exists():raise ValueError('Do not overwrite a frozen expansion')
 images=np.stack([array(r) for r in rows]);path=ROOT/'targets.mmap';images.tofile(path)
 records=[]
 for i,r in enumerate(rows):
  r['target']={'path':str(path.resolve()),'shape':list(images.shape),'index':i}
  r['filename_text']=r['short'];r['descriptive_text']=r['long']
  for kind in r['caption_variants']:
   assert r[kind].strip() and '_' not in r[kind]
   records.append({**r,'id':r['id']+':'+kind,'prompt':r[kind],'caption_variant':kind,'approved_for_training':False,'caption_supervision':'experimental_automatic_weak_label'})
 atomic_json(ROOT/'cases.json',rows)
 meta=write_catalog(ROOT/'catalog',records,{'native_size':16,'annotation_sha256':file_hash(ROOT/'annotations.jsonl'),'caption_quality_review_sha256':file_hash(ROOT/'caption_quality_review.json'),'input_sha256':file_hash(ROOT/'annotation_cases.jsonl'),'core_cases_sha256':file_hash(CORE/'cases.json'),'full_visual_review_sha256':file_hash(ROOT/'full_visual_review.json'),'scope':'Experimental 15-category expansion. Original project/content splits; filename always retained; eligible descriptions sampled 50/50, rejected descriptions fall back to filename only; target-first uniform sampling. Image completeness visually reviewed; captions not fully manually validated.'})
 cfg=yaml.safe_load((CORE/'config.yaml').read_text());cfg['train'].update(steps=6000,lr=5e-5,warmup=100,eval_every=500,val_samples=256,seed=20261116)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 config=Experiment.load(ROOT/'config.yaml');ds=CatalogDataset(records,16,128,'train',asset_type='item',task='text');sampler=TaskSampler(ds,config)
 indices,modes=sampler.sample(100000,torch.Generator().manual_seed(20261116));counts=Counter(ds.records[i]['caption_variant'] for i,m in zip(indices,modes) if m=='text')
 fraction=counts['short']/sum(counts.values())
 group_sizes={ds.records[g[0]]['target_hash']:len(g) for g in sampler.pools['text']}
 assert len(group_sizes)==sum(r['split']=='train' for r in rows)
 assert set(group_sizes.values()) <= {1,2}
 dual_counts=Counter(ds.records[i]['caption_variant'] for i,m in zip(indices,modes) if m=='text' and group_sizes[ds.records[i]['target_hash']]==2)
 dual_fraction=dual_counts['short']/sum(dual_counts.values());assert .49<dual_fraction<.51
 expected_short=sum(1/size for size in group_sizes.values())/len(group_sizes)
 assert abs(fraction-expected_short)<.01
 atomic_json(ROOT/'sampling_audit.json',{'draws':len(indices),'mode_counts':dict(Counter(modes)),'text_caption_counts':dict(counts),'filename_fraction_given_text':fraction,'expected_filename_fraction':expected_short,'dual_caption_filename_fraction':dual_fraction,'dual_caption_counts':dict(dual_counts),'unique_train_targets':len(group_sizes),'caption_count_per_target':dict(Counter(group_sizes.values())),'training_exposure':sampler.report(config)})
 # New holdout selection depends only on immutable candidate IDs, never annotation quality or outputs.
 candidate_val=[r for r in inputs if r['split']=='val'];random.Random(20261115).shuffle(candidate_val)
 chosen_ids=[r['id'] for r in candidate_val[:32]];val_byid={r['id']:r for r in new_val}
 evaluation=json.loads((CORE/'eval_cases.json').read_text());cases=evaluation['cases'];missing=[]
 for rid in chosen_ids:
  if rid not in val_byid:missing.append(rid);continue
  r=val_byid[rid]
  for kind in r['caption_variants']:cases.append({**r,'evaluation_prompt':r[kind],'group':'expansion_val_'+kind,'seed':20261117})
 atomic_json(ROOT/'eval_cases.json',{**evaluation,'cases':cases,'new_holdout_ids_before_caption_filter':chosen_ids,'omitted_excluded_cases':missing,'note':'Prior 171 probes retained verbatim; new validation IDs deterministically selected independently of label quality. Development evaluation only; no test used.'})
 init=CORE/'training/latest.pt'
 atomic_json(ROOT/'protocol.json',{'initialization':str(init),'initialization_sha256':file_hash(init),'config_sha256':file_hash(ROOT/'config.yaml'),'annotation_sha256':file_hash(ROOT/'annotations.jsonl'),'core_targets':len(core),'new_targets':len(selected),'train_targets':sum(r['split']=='train' for r in rows),'val_targets':sum(r['split']=='val' for r in rows),'excluded':excluded,'filename_only_targets':sum(len(r['caption_variants'])==1 for r in rows),'rejected_descriptions':[{'id':r['id'],'split':r['split'],'reason':r['long_caption_rejection']} for r in rows if r['long_caption_rejection']],'category_counts':dict(Counter(r['category'] for r in rows)),'endpoint':6000,'caption_policy':'Uniform target; two valid captions sampled 50/50, otherwise filename only. Bad descriptions do not remove valid images. Unconditional mode remains 10%; text mode 90%.','test_used':False,'production_promoted':False,'limitations':'Automatic weak captions; all candidate images reviewed for completeness, not all captions certified; conservative filename filtering; only 15 categories; experiment changes scale, exposure and learning rate together.'})
 print(json.dumps({'train':sum(r['split']=='train' for r in rows),'val':sum(r['split']=='val' for r in rows),'excluded_images_after_initial_screen':len(excluded),'sampling_filename_fraction':fraction}),flush=True)
if __name__=='__main__':main()
