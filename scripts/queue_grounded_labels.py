"""Run resumable train-only label revision shards and export an auditable result."""
import os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json
ROOT=Path('outputs/grounded_label_revision_10k_20261103')
if __name__=='__main__':
 processes=[];logs=[]
 try:
  for shard in [0,1]:
   log=(ROOT/f'full{shard}.log').open('a');logs.append(log)
   p=subprocess.Popen([sys.executable,'-u','scripts/annotate_grounded_selection.py','--device',f'cuda:{shard}','--shard',str(shard),'--shards','2','--batch-size','8','--tag',f'full{shard}'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'})
   processes.append(p)
  atomic_json(ROOT/'queue_status.json',{'stage':'relabeling_existing_10000_descriptions','pids':[p.pid for p in processes]})
  while any(p.poll() is None for p in processes):time.sleep(15)
  if any(p.returncode for p in processes):raise RuntimeError(f'Shard failures: {[p.returncode for p in processes]}')
  subprocess.run([sys.executable,'scripts/export_grounded_labels.py','--root',str(ROOT)],check=True)
  atomic_json(ROOT/'queue_status.json',{'stage':'complete','training_started':False})
 except BaseException as e:
  for p in processes:
   if p.poll() is None:p.terminate()
  atomic_json(ROOT/'queue_status.json',{'stage':'failed','error':str(e)})
  raise
 finally:
  for log in logs:log.close()
