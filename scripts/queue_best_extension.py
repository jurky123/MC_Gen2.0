"""Append the user-requested minimum-error checkpoint comparison after final eval."""
import json,os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json
ROOT=Path('outputs/text_expansion22k_conservative_20261102')
if __name__=='__main__':
 while True:
  status=json.loads((ROOT/'status.json').read_text())
  if status['stage']=='failed':raise SystemExit('Main run failed')
  if status['stage']=='complete_pending_visual_review':break
  time.sleep(30)
 try:
  for stage,args in [('best_evaluation',['scripts/evaluate_text_recovery.py','--root',str(ROOT),'--best-flow','--device','cuda:1']),('best_summary',['scripts/summarize_text_recovery.py','--root',str(ROOT),'--best-flow']),('checkpoint_comparison',['scripts/compare_extension_checkpoints.py'])]:
   atomic_json(ROOT/'status.json',{'stage':stage,'updated_unix':time.time()})
   with (ROOT/(stage+'.log')).open('a') as log:subprocess.run([sys.executable,*args],env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
  atomic_json(ROOT/'status.json',{'stage':'complete_pending_visual_review','best_and_final_compared':True,'step':22000,'production_promoted':False})
 except Exception as exc:atomic_json(ROOT/'status.json',{'stage':'failed','error':repr(exc)});raise
