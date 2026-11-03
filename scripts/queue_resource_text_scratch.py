"""Run one new random-initialized experiment without starting the abandoned continuation."""
import os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/resource_text_scratch_20261103')
if __name__=='__main__':
 try:
  atomic_json(ROOT/'queue_identity.json',{'driver_sha256':file_hash('scripts/run_resource_text_scratch.py'),'config_sha256':file_hash(ROOT/'config.yaml'),'catalog_manifest_sha256':file_hash(ROOT/'catalog/samples.jsonl'),'initialize':None,'generator':'random weights','optimizer':'fresh AdamW','qwen':'frozen pretrained text encoder; cached features reused'})
  for stage in ['smoke','train','evaluate']:
   atomic_json(ROOT/'queue_status.json',{'stage':stage,'updated_unix':time.time()})
   with (ROOT/f'{stage}.log').open('a') as log:
    subprocess.run([sys.executable,'-u','scripts/run_resource_text_scratch.py',stage],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'},check=True)
  atomic_json(ROOT/'queue_status.json',{'stage':'complete','updated_unix':time.time()})
 except BaseException as e:
  atomic_json(ROOT/'queue_status.json',{'stage':'stopped' if isinstance(e,subprocess.CalledProcessError) and e.returncode==2 else 'failed','error':str(e),'updated_unix':time.time()});raise
