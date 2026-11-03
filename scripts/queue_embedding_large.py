"""Download readiness -> official embedding cache / concurrent batch tuning -> scratch training."""
import json,os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/qwen_vl_embedding_large_20261103');MODEL=Path('/home/iflab/models/Qwen3-VL-Embedding-8B')
def status(stage,**extra):atomic_json(ROOT/'queue_status.json',{'stage':stage,'updated_unix':time.time(),**extra});print(stage,extra,flush=True)
def command(stage,script,*args):
 status(stage)
 with (ROOT/f'{stage}.log').open('a') as log:subprocess.run([sys.executable,'-u',script,*args],stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'})
if __name__=='__main__':
 cache=None
 try:
  status('waiting_for_verified_model')
  while not (MODEL/'verified_download.json').exists():time.sleep(10)
  status('encoding_and_batch_probe')
  log=(ROOT/'features.log').open('a');cache=subprocess.Popen([sys.executable,'-u','scripts/cache_qwen_vl_embedding.py'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'})
  while not (ROOT/'probe_data.pt').exists():
   if cache.poll() is not None:raise RuntimeError('Embedding smoke failed; inspect features.log')
   time.sleep(5)
  command('batch_tuning','scripts/tune_embedding_batch.py')
  status('finishing_embedding_cache');code=cache.wait();log.close()
  if code:raise RuntimeError(f'Embedding cache failed:{code}')
  atomic_json(ROOT/'run_identity.json',{'generator':'random','model_revision':json.load(open(MODEL/'verified_download.json'))['revision'],'config_sha256':file_hash(ROOT/'config.yaml'),'model_code_sha256':file_hash('src/mcgen/model.py'),'embedding_code_sha256':file_hash('src/mcgen/qwen_embedding.py'),'train_driver_sha256':file_hash('scripts/run_embedding_large.py'),'budget':json.load(open(ROOT/'batch_and_budget.json'))})
  for stage in ['smoke','train','evaluate']:command(stage,'scripts/run_embedding_large.py',stage)
  status('complete')
 except BaseException as e:
  if cache is not None and cache.poll() is None:cache.terminate()
  status('failed',error=str(e));raise
