"""Freeze candidate exclusions and a pre-caption audit sample; do not claim full review."""
import json,re
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 rows=json.loads((ROOT/'candidates.json').read_text())
 manual={75:'isolated pixel instead of complete pickaxe',1470:'fishing bobber asset',1956:'drink instead of apple',1942:'socket instead of gem',1952:'amulet component',1822:'sword glow overlay'}
 tokens={'bobber','boba','socket','amulet','glow','inner'}
 excluded=[];kept=[]
 for r in rows:
  matched=sorted(set(re.sub('[^a-z]+',' ',r['source_filename'].lower()).split())&tokens)
  reason=manual.get(r['review_index']) or ('component_or_noncategory_filename:'+','.join(matched) if matched else None)
  if reason:excluded.append({'id':r['id'],'index':r['review_index'],'reason':reason})
  else:kept.append(r)
 path=ROOT/'annotation_cases.jsonl'
 if path.exists():raise ValueError('Frozen input already exists')
 path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
 atomic_json(ROOT/'visual_review.json',{'scope':'Assistant visually inspected first 96 candidates and fixed random 64 sample (48 train,16 validation), not all candidates. Completeness/category screening only. Automatic descriptions remain weak labels.','reviewed_indices':sorted(set(range(96))|set(json.loads((ROOT/'fixed_review64.json').read_text())['indices'])),'additional_excluded_filename_tokens':sorted(tokens),'excluded':excluded,'kept':len(kept),'input_sha256':file_hash(path)})
 print({'kept':len(kept),'excluded':len(excluded)})
