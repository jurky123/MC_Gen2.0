"""Build a filename-only revision preserving exact pixels, hashes and split ownership."""
import io,json,random,zipfile
from pathlib import Path
from collections import defaultdict,Counter
import pyarrow.parquet as pq
import yaml
from PIL import Image
from mcgen.catalog import atomic_json,file_hash,write_catalog
from mcgen.images import content_hash
from mcgen.resource_text import extract_resource_text,POLICY_VERSION
ROOT=Path('outputs/resource_text_training_20261103')
SOURCE=Path('outputs/fullscale_text10k_20261103')
PARENT=SOURCE/'training/best_flow.pt'

def build():
 ROOT.mkdir(exist_ok=False);index=defaultdict(list)
 archives=json.load(open('outputs/training_label_audit_20261103/archive_audit.json'))
 for url,archive in {r['url']:r['archive_file'] for r in archives}.items():
  with zipfile.ZipFile(archive) as z:
   for name in z.namelist():
    if not name.endswith('.png'):continue
    try:
     image=Image.open(io.BytesIO(z.read(name))).convert('RGBA')
     if image.size==(16,16):index[(url,Path(name).name,content_hash(image))].append(name)
    except (OSError,ValueError):continue
 urls=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=['version_url'])['version_url'].to_pylist()
 records=[];counts=Counter();reasons=Counter();changed=Counter();category=Counter();review=[]
 with (ROOT/'decisions.jsonl').open('w') as decisions:
  for line in (SOURCE/'catalog/samples.jsonl').open():
   r=json.loads(line)
   if r['caption_variant']!='short':continue
   assert r['split'] in ('train','val')
   paths=index[(urls[r['source_row']],r['source_filename'],r['target_hash'])]
   result=extract_resource_text(r['source_filename'],r.get('source_mod',''),paths,r['visible_pixels'])
   counts[r['split']+'/'+result['decision']]+=1;reasons.update(result['reasons'])
   decisions.write(json.dumps({'target_hash':r['target_hash'],'split':r['split'],'source_filename':r['source_filename'],'old_text':r['prompt'],**result},ensure_ascii=False)+'\n')
   if result['decision']!='keep':continue
   prompt=result['text'];assert prompt and '_' not in prompt
   changed[r['split']]+=int(prompt!=r['prompt'])
   new={**r,'id':'resource:'+r['target_hash']+':short','prompt':prompt,'filename_text':prompt,
        'caption_variant':'short','caption_supervision':'deterministic_resource_text_only','resource_paths':paths,
        'resource_text_policy':POLICY_VERSION,'resource_path_verified':bool(paths),'removed_technical_tokens':result['removed_technical_tokens']}
   records.append(new)
   if r['split']=='train':category[r.get('category_15','other')]+=1
 write_catalog(ROOT/'catalog',records,{'native_size':16,'parent_catalog':str(SOURCE/'catalog'),'source_manifest_sha256':file_hash(SOURCE/'catalog/samples.jsonl'),'policy_sha256':file_hash('src/mcgen/resource_text.py'),'supervision':'File information only; no VLM labels or VLM rejection decisions. Exact targets and original split groups retained.','visual_descriptions_used':False,'test_used':False})
 cfg=yaml.safe_load((SOURCE/'config.yaml').read_text());cfg['train'].update(steps=10000,lr=2e-6,warmup=500,eval_every=1000,val_samples=1024,seed=2026110308,schedule_start=0)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 # Fixed development probes: novel prompts plus broad validation, never test images.
 cases=json.load(open('outputs/fullscale_eval6000_20261103/eval_cases.json'))['cases']
 chosen=[r for r in cases if r['group']=='novel_composition']
 chosen += [r for r in cases if r['group']=='fullscale_val_filename'][:32]
 atomic_json(ROOT/'eval_cases.json',chosen)
 atomic_json(ROOT/'build_report.json',{'counts':dict(counts),'quarantine_reasons':dict(reasons),'changed_text_targets':dict(changed),'train_categories':dict(category),'parent_checkpoint':str(PARENT),'parent_sha256':file_hash(PARENT),'all_caption_variants':'short; one per target','visual_descriptions_used':False})
 # Fresh before/after sample, including changed names and reclaimed previously VLM-rejected items.
 rng=random.Random(2026110308);train=[r for r in records if r['split']=='train']
 changedrows=[r for r in train if r['removed_technical_tokens']]
 selected=rng.sample(train,24)+rng.sample(changedrows,min(8,len(changedrows)))
 atomic_json(ROOT/'sample.json',selected);print(json.dumps(json.load(open(ROOT/'build_report.json')),indent=2),flush=True)
if __name__=='__main__':build()
