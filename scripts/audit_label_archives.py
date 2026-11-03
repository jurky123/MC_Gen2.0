"""Read-only filename and RGBA checks against locally downloaded original Mod archives."""
import json,hashlib,zipfile,io
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('outputs/training_label_audit_20261103');rows=json.load(open(root/'sample.json'));dest=root/'source_archives';dest.mkdir(exist_ok=True)
selected=rows[:4]+[r for r in rows if r['group']=='diagnostic']
urls=sorted({r['raw_metadata']['version_url'] for r in selected})
files={}
for u in urls:
 p=dest/(hashlib.sha256(u.encode()).hexdigest()[:16]+'.jar')
 files[u]=(p,None) if p.exists() else (None,'Archive not downloaded')
results=[]
for r in selected:
 url=r['raw_metadata']['version_url'];p,error=files[url];report={'sample_id':r['sample_id'],'source_filename':r['source_filename'],'url':url,'download_error':error,'basename_matches':[],'pixel_matches':[], 'visible_rgba_matches':[]}
 if p:
  report['archive_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();report['archive_file']=str(p)
  target=np.array(Image.open(root/r['image']).convert('RGBA'))
  with zipfile.ZipFile(p) as z:
   names=[n for n in z.namelist() if n.lower().endswith('.png')]
   hits=[n for n in names if Path(n).name==r['source_filename']];report['basename_matches']=hits
   # Exact names first; if no exact-name pixel match, scan other PNGs to detect renamed assets.
   for name in hits+[n for n in names if n not in hits]:
    try:
     image=np.array(Image.open(io.BytesIO(z.read(name))).convert('RGBA'))
     if image.shape==target.shape:
      if np.array_equal(image,target):report['pixel_matches'].append({'path':name,'match':'exact_rgba'})
      if np.array_equal(image[:,:,3],target[:,:,3]) and np.array_equal(image[:,:,:3][target[:,:,3]>0],target[:,:,:3][target[:,:,3]>0]):report['visible_rgba_matches'].append({'path':name,'match':'alpha_and_all_visible_rgb_exact', 'hidden_rgb_differing_channels':int(np.count_nonzero(image[:,:,:3][target[:,:,3]==0]!=target[:,:,:3][target[:,:,3]==0]))})
    except Exception:pass
   report['png_count']=len(names)
 results.append(report)
 print(json.dumps(report,ensure_ascii=False),flush=True)
(root/'archive_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
