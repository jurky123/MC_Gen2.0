"""Record the completed fixed sample review without declaring weak labels gold."""
import json
from pathlib import Path
from collections import Counter
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 partial=json.loads((ROOT/'caption_review_partial.json').read_text());sample=json.loads((ROOT/'fixed_review64.json').read_text())['indices'];rows=json.loads((ROOT/'candidates.json').read_text());inputs={r['id'] for r in map(json.loads,(ROOT/'annotation_cases.jsonl').read_text().splitlines())};labels=[json.loads(s) for s in (ROOT/'annotations.jsonl').read_text().splitlines()]
 assert len(labels)==len(inputs)==1983
 early={r['index']:r['reason'] for r in json.loads((ROOT/'visual_review.json').read_text())['excluded']};ratings=[]
 for i in sample:
  r=rows[i]
  if r['id'] not in inputs:rating='image_excluded_before_annotation';reason=early[i]
  else:
   assert i in partial['reviewed_indices']
   if str(i) in partial.get('clear_contradictions',{}):rating='clear_contradiction';reason=partial['clear_contradictions'][str(i)]
   elif str(i) in partial.get('unsupported_details',{}):rating='unsupported_detail';reason=partial['unsupported_details'][str(i)]
   else:rating='plausible';reason='No clear major identity/color/part contradiction in this visual review; not a gold annotation.'
  ratings.append({'index':i,'id':r['id'],'split':r['split'],'rating':rating,'reason':reason})
 atomic_json(ROOT/'caption_quality_review.json',{'scope':'Fixed 64 candidates selected before captions: 58 image-caption pairs inspected, 6 images excluded before annotation. Assistant nonblind qualitative review, not expert accuracy. All descriptive labels remain weak.','annotation_sha256':file_hash(ROOT/'annotations.jsonl'),'ratings':ratings,'counts':dict(Counter(r['rating'] for r in ratings)),'decision':'Proceed with authorized expansion. Retain valid images with filename-only supervision when description is unreliable, as requested by user.','user_policy':'Bad description drops only descriptive caption; target-uniform sampling remains unchanged.'})
 print(dict(Counter(r['rating'] for r in ratings)))
