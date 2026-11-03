"""Frozen label sample and provenance audit; never generates replacement captions."""
import io,json,random,hashlib,csv
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from PIL import Image
from mcgen.images import content_hash,native_grid
from mcgen.semantic_captions import filename_words

ROOT=Path('outputs/training_label_audit_20261103')
RUN=Path('outputs/fullscale_text10k_20261103')
RAW=Path('data/raw/mc_text2image/data')
FIELDS=['file_name','project_id','mod_slug','version_url','type']

def main():
 ROOT.mkdir(exist_ok=False);(ROOT/'images').mkdir();(ROOT/'raw_images').mkdir()
 meta=pq.read_table('data/build/mc_text2image32_wl/metadata.parquet',columns=FIELDS+['index'])
 assert np.array_equal(meta['index'].to_numpy(),np.arange(len(meta)))
 names=meta['file_name'].to_pylist();mods=meta['mod_slug'].to_pylist()
 long={};counts=Counter()
 for line in (RUN/'catalog/samples.jsonl').open():
  r=json.loads(line)
  if r['split']=='train' and r['caption_variant']=='long':long[r['target_hash']]=r
 rng=random.Random(20261103);pools={'dual':[],'filename_only':[]};n=Counter();audit=Counter();mismatches=[]
 for line in (RUN/'catalog/samples.jsonl').open():
  r=json.loads(line)
  if r['split']!='train' or r['caption_variant']!='short':continue
  i=r['source_row'];audit['train_targets']+=1
  for key,ok in [('filename_matches_metadata',r['source_filename']==names[i]),('mod_matches_metadata',r['source_mod']==mods[i]),('short_matches_filename_extraction',r['prompt']==' '.join(filename_words(r['source_filename'])))]:
   audit[key]+=int(ok)
   if not ok:mismatches.append({'row':i,'field':key,'record':r})
  audit['multiple_filename_aliases']+=int(len({a[0] for a in r.get('source_aliases',[])})>1)
  g='dual' if r['target_hash'] in long else 'filename_only';n[g]+=1;limit=48 if g=='dual' else 16
  if len(pools[g])<limit:pools[g].append(r)
  else:
   j=rng.randrange(n[g])
   if j<limit:pools[g][j]=r
 selected=[(g,r) for g in pools for r in pools[g]]
 # Separate diagnostic probes: not presented as a random estimate of corpus quality.
 probes=json.load(open('outputs/pixel_quality_pilot_20261103/candidate_groups.json'))['control']
 selected += [('diagnostic',probes[i]) for i in [10,73,74,76,78,82,87,107]]
 annotations={json.loads(s)['target_hash']:json.loads(s) for s in (RUN/'annotations.jsonl').open()}
 retained={r['target_hash']:r for r in json.load(open('outputs/text_expansion2k_20261102/cases.json'))}
 rows=[];maps={}
 for j,(group,r) in enumerate(selected):
  h=r['target_hash'];a=annotations.get(h);older=retained.get(h);desc=long.get(h)
  if desc:status='已用于训练'
  elif a:status='生成过，未作为长标签用于训练'
  else:status='当前训练仅使用文件名标签；本批标注记录中无自动描述'
  s=r['target'];key=s['path']
  if key not in maps:maps[key]=np.memmap(key,mode='r',dtype='uint8',shape=tuple(s['shape']))
  pixels=np.array(maps[key][s['index']]);assert content_hash(pixels)==h
  Image.fromarray(pixels).save(ROOT/'images'/f'{j:03d}.png')
  m=meta.slice(r['source_row'],1).to_pylist()[0]
  rows.append({'sample_id':j,'group':group,'target_hash':h,'source_row':r['source_row'],'source_filename':r['source_filename'],'filename_label':r['prompt'],'automatic_training_label':desc['prompt'] if desc else None,'automatic_status':status,'raw_automatic_candidate':a.get('combined_text') if a else None,'visual_description':a.get('visual_text') if a else (older.get('visual_text') if older else None),'quality_flags':a.get('quality_flags') if a else None,'caption_supervision':desc.get('caption_supervision') if desc else r.get('caption_supervision'),'source_aliases':r.get('source_aliases',[]),'raw_metadata':m,'image':f'images/{j:03d}.png'})
 wanted={r['source_row'] for r in rows};byindex={}
 for r in rows:byindex.setdefault(r['source_row'],[]).append(r)
 offset=0;rawaudit=Counter();fieldmismatches=[]
 for path in sorted(RAW.glob('*.parquet')):
  pf=pq.ParquetFile(path);local=0
  for batch in pf.iter_batches(batch_size=8192,columns=FIELDS):
   for f in FIELDS:
    equal=pc.fill_null(pc.equal(batch.column(f),meta[f].slice(offset+local,len(batch))),False)
    # Both null fields are equal.
    equal=pc.or_(equal,pc.and_(pc.is_null(batch.column(f)),pc.is_null(meta[f].slice(offset+local,len(batch)))))
    bad=len(batch)-pc.sum(pc.cast(equal,'int64')).as_py();rawaudit[f+'_mismatches']+=bad
    if bad:fieldmismatches.append({'file':str(path),'start':offset+local,'field':f,'count':bad})
   local+=len(batch)
  groupoff=0
  for gi in range(pf.num_row_groups):
   count=pf.metadata.row_group(gi).num_rows;hits=sorted(i for i in wanted if offset+groupoff<=i<offset+groupoff+count)
   if hits:
    t=pf.read_row_group(gi,columns=['image']+FIELDS+['text'])
    for idx in hits:
     src=t.slice(idx-offset-groupoff,1).to_pylist()[0];cell=src['image'];blob=cell['bytes'];im=Image.open(io.BytesIO(blob)).convert('RGBA');rawpixels=np.array(im)
     im.save(ROOT/'raw_images'/f'{idx}.png')
     for row in byindex[idx]:
      expected=np.array(Image.open(ROOT/row['image']));row['raw_parquet']={'file':str(path),'row_in_file':idx-offset,'image_path':cell.get('path'),'original_text':src['text'],'image_size':list(im.size),'image_bytes_sha256':hashlib.sha256(blob).hexdigest(),'filename':src['file_name'],'pixels_match_training':bool(np.array_equal(native_grid(rawpixels,16),expected))}
   groupoff+=count
  offset+=pf.metadata.num_rows
 assert offset==len(meta)
 report={'seed':20261103,'sampling':'48 uniform target samples from dual-caption training pool +16 from filename-only pool; 8 separately identified diagnostic probes. Not a frequency-representative sample. No new labels generated.','pool_counts':dict(n),'training_audit':dict(audit),'training_mismatches':mismatches,'raw_rows_checked':offset,'raw_metadata_audit':dict(rawaudit),'raw_metadata_mismatches':fieldmismatches,'sample_pixel_checks':{'checked':len(rows),'passed':sum(r['raw_parquet']['pixels_match_training'] for r in rows)}}
 (ROOT/'sample.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(ROOT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 with (ROOT/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['sample_id','group','source_filename','filename_label','automatic_training_label','automatic_status','raw_automatic_candidate','source_row','target_hash'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
