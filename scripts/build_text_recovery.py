"""Freeze the reviewed core, retaining original splits and weak-label provenance."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from mcgen.catalog import atomic_json,write_catalog,file_hash
from mcgen.semantic_captions import filename_words
from audit_item_captions import array

ROOT=Path('outputs/text_recovery_core_20261102')
# Recorded from all 10 candidate sheets BEFORE recovery training.
GROUPS={
 'partial_asset_or_overlay':[26,64,86,88,126,133,145,170,188,210,268,269,270,272,273,274,275,281,283,284,287,288,289,290,291,293,302,353,398,407,417,419,458],
 'ui_or_mold_or_background':[6,7,162,464],
 'different_or_ambiguous_category':[44,57,74,111,120,181,278,355,356,358,364,367,374,377,387,388,459,466,468],
}
if __name__=='__main__':
 rows=json.loads((ROOT/'candidates.json').read_text());reasons={i:k for k,indices in GROUPS.items() for i in indices}
 selected=[r for r in rows if r['review_index'] not in reasons]
 atomic_json(ROOT/'visual_review.json',{'reviewer':'assistant_visual_review','scope':'All 477 candidate images and filenames inspected before training. Completeness/category review only; long captions remain automatic weak labels, not fully certified.','excluded':[{'index':r['review_index'],'id':r['id'],'reason':reasons[r['review_index']]} for r in rows if r['review_index'] in reasons],'kept':len(selected),'candidate_sha256':file_hash(ROOT/'candidates.json')})
 images=np.stack([array(r) for r in selected]);path=ROOT/'targets.mmap'
 if path.exists():raise ValueError('Do not overwrite prepared recovery data')
 images.tofile(path)
 for i,r in enumerate(selected):
  r['target']={'path':str(path.resolve()),'shape':list(images.shape),'index':i}
  r['short']=' '.join(filename_words(r['source_filename']))
 atomic_json(ROOT/'cases.json',selected)
 records=[]
 for r in selected:
  for kind in (['short','long'] if r['split']=='train' else ['short']):
   records.append({**r,'id':r['id']+':'+kind,'prompt':r[kind],'caption_variant':kind,'caption_supervision':'automatic_weak_label_with_visual_completeness_review'})
 write_catalog(ROOT/'catalog',records,{'native_size':16,'review_sha256':file_hash(ROOT/'visual_review.json'),'scope':'Complete-item core recovery; unchanged original project/content splits. Two prompt variants, target-uniform sampling. Not a full-domain catalog.'})
 cfg=yaml.safe_load(Path('configs/v3/experiments/text_caption_ab1152.yaml').read_text());cfg['train'].update(steps=4000,lr=1e-4,warmup=100,eval_every=500,val_samples=128,seed=20261111)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 counts=Counter((r['split'],r['category']) for r in selected)
 init='outputs/text_caption_scale1152_20261102/training/mixed/latest.pt'
 atomic_json(ROOT/'protocol.json',{'initialization':init,'initialization_sha256':file_hash(init),'config_sha256':file_hash(ROOT/'config.yaml'),'counts':{f'{s}:{c}':n for (s,c),n in counts.items()},'train_targets':sum(r['split']=='train' for r in selected),'val_targets':sum(r['split']=='val' for r in selected),'test_used':False,'endpoint':4000,'evaluation':'Compare fixed endpoint to initialization on core validation, prior broad 64-name/12-rich cases, 16 train fit probes, and predefined novel color/category prompts. Fixed seeds/32 Heun/CFG2. Core validation is a selected development subset, not independent test.','attribution':'Recipe experiment changes data subset, learning rate and exposure together. Not an isolated data-cleaning ablation.','promotion':'No automatic production promotion based on flow loss.'})
 print(json.dumps({'train':sum(r['split']=='train' for r in selected),'val':sum(r['split']=='val' for r in selected),'excluded':len(reasons)},indent=2))
