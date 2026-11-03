"""Revalidate independent pilot observations and stage the agreed 10k revision scope."""
import json,hashlib
from pathlib import Path
from collections import Counter
from mcgen.catalog import atomic_json,file_hash
from mcgen.grounded_labels import decide
SOURCE=Path('outputs/grounded_label_revision_20261103')
DEST=Path('outputs/grounded_label_revision_10k_20261103')
def seal(p):
 p.pop('signature',None);p['signature']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();return p
if __name__=='__main__':
 protocol=json.load(open(SOURCE/'protocol.json'));atomic_json(SOURCE/'pilot_observation_protocol.json',protocol)
 path=SOURCE/'labels_pilot.jsonl';observed=[json.loads(s) for s in path.open()];atomic_json(SOURCE/'pilot_original_observations.json',observed)
 protocol['policy_sha256']=file_hash('src/mcgen/grounded_labels.py');seal(protocol);atomic_json(SOURCE/'protocol.json',protocol)
 corrected=[]
 for r in observed:
  previous=r['signature']
  if r.get('visual_observation') and r.get('semantic_review'):r.update(decide(r,r['visual_observation'],r['semantic_review']))
  r.update(observation_source_signature=previous,signature=protocol['signature']);corrected.append(r)
 with path.open('w') as out:
  for r in corrected:out.write(json.dumps(r,ensure_ascii=False)+'\n')
 DEST.mkdir(exist_ok=False);sample_hashes={r['target_hash'] for r in corrected};count=0;counts=Counter()
 with (DEST/'cases.jsonl').open('w') as subset,(SOURCE/'file_label_inventory.jsonl').open('w') as inv:
  for line in (SOURCE/'cases.jsonl').open():
   r=json.loads(line);reject=r['resource_flags']+(['insufficient_visible_content'] if r['visible_pixels']<8 else [])
   state='quarantine' if reject else 'filename_candidate_pending_independent_review'
   inv.write(json.dumps({'target_hash':r['target_hash'],'source_filename':r['source_filename'],'raw_filename_label':r['old_filename_label'],'file_label':None if reject else r['old_filename_label'],'status':state,'resource_paths':r['resource_paths'],'path_status':r['path_status'],'reasons':reject},ensure_ascii=False)+'\n');counts[state]+=1
   if r['old_visual_label'] or r['target_hash'] in sample_hashes:subset.write(line);count+=1
 p={**protocol,'cases_sha256':file_hash(DEST/'cases.jsonl'),'total':count,'scope':'All10000 existing automatic-caption training targets plus72 diagnostic examples (deduplicated). No new full-corpus visual annotation.','parent_case_manifest_sha256':protocol['cases_sha256'],'reused_pilot_observations':72};seal(p);atomic_json(DEST/'protocol.json',p)
 with (DEST/'labels_pilot.jsonl').open('w') as out:
  for r in corrected:r.update(observation_source_signature=r['signature'],signature=p['signature']);out.write(json.dumps(r,ensure_ascii=False)+'\n')
 atomic_json(SOURCE/'file_label_inventory_summary.json',{'total':sum(counts.values()),'decisions':dict(counts),'note':'Unresolved paths are explicitly unverified. Candidate labels are not certified or silently promoted. Per-image independent revision follows for the agreed10k scope.'})
 print('staged',count,'revision targets; inventory',dict(counts))
