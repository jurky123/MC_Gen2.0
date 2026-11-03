"""Run the authorized filename-only revision through cached training and fixed evaluation."""
import os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/resource_text_training_20261103')
if __name__=='__main__':
 try:
  atomic_json(ROOT/'queue_identity.json',{'driver_sha256':file_hash('scripts/run_resource_text_training.py'),'extractor_sha256':file_hash('src/mcgen/resource_text.py'),'config_sha256':file_hash(ROOT/'config.yaml'),'initialize':'accepted fullscale best2000','updates':10000,'supervision':'file information only','optimizer':'Fresh Adam,500 warmup updates; initialized model weights from best2000.'})
  for stage in ['features','smoke','train','evaluate']:
   atomic_json(ROOT/'queue_status.json',{'stage':stage,'updated_unix':time.time()})
   with (ROOT/f'{stage}.log').open('a') as log:
    subprocess.run([sys.executable,'-u','scripts/run_resource_text_training.py',stage],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'},check=True)
  atomic_json(ROOT/'queue_status.json',{'stage':'complete','updated_unix':time.time()})
 except BaseException as e:
  atomic_json(ROOT/'queue_status.json',{'stage':'failed','error':str(e),'updated_unix':time.time()});raise
