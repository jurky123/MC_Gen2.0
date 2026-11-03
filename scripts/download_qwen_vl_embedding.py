"""Download the pinned official embedding checkpoint and verify every artifact."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi,hf_hub_download
from mcgen.catalog import atomic_json
REPO='Qwen/Qwen3-VL-Embedding-8B'
REVISION='2c4565515e0f265c6511776e7193b22c0968ddc7'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/home/iflab/models/Qwen3-VL-Embedding-8B'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True);entries=[]
 for s in info.siblings:
  if not s.rfilename.startswith('.') and '/' not in s.rfilename and s.rfilename.endswith(('.json','.txt','.model','.safetensors','.jinja')):entries.append({'name':s.rfilename,'size':s.size,'sha256':s.lfs.sha256 if s.lfs else None,'blob_id':s.blob_id})
 atomic_json(a.out/'download_manifest.json',{'repo':REPO,'revision':REVISION,'files':entries})
 def download(e):
  f=a.out/e['name']
  if not f.exists() or f.stat().st_size!=e['size']:hf_hub_download(REPO,e['name'],revision=REVISION,local_dir=a.out)
 list(ThreadPoolExecutor(max_workers=4).map(download,entries));checked=[]
 for e in entries:
  f=a.out/e['name'];assert f.stat().st_size==e['size'];h=hashlib.sha256();g=hashlib.sha1(('blob '+str(e['size'])+'\0').encode())
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(8<<20),b''):h.update(b);g.update(b)
  assert h.hexdigest()==e['sha256'] if e['sha256'] else g.hexdigest()==e['blob_id']
  checked.append({**e,'verified_sha256':h.hexdigest()})
 atomic_json(a.out/'verified_download.json',{'repo':REPO,'revision':REVISION,'files':checked,'complete':True});print('Verified',REPO,REVISION)
