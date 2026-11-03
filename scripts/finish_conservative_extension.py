"""Finish fixed evaluation after conservative training; fail if training exits early."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json
ROOT=Path('outputs/text_expansion22k_conservative_20261102')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int,required=True);a=p.parse_args()
 while True:
  s=json.loads((ROOT/'status.json').read_text())
  if s['stage']=='training_complete':break
  if s['stage']=='failed':raise SystemExit('Training failed')
  try:os.kill(a.training_pid,0)
  except ProcessLookupError:
   atomic_json(ROOT/'status.json',{'stage':'failed','error':'Training process exited before completion; checkpoints preserved'});raise SystemExit(1)
  time.sleep(30)
 stages=[('evaluation',['scripts/evaluate_text_recovery.py','--root',str(ROOT),'--device','cuda:1']),('summary',['scripts/summarize_text_recovery.py','--root',str(ROOT)]),('export',['scripts/export_text_expansion_evaluation.py','--root',str(ROOT)]),('fit_diagnosis',['scripts/diagnose_expansion_fit.py','--extended-root',str(ROOT)])]
 try:
  for stage,args in stages:
   atomic_json(ROOT/'status.json',{'stage':stage,'updated_unix':time.time()})
   with (ROOT/(stage+'.log')).open('a') as log:subprocess.run([sys.executable,*args],env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
  atomic_json(ROOT/'status.json',{'stage':'complete_pending_visual_review','step':22000,'production_promoted':False})
 except Exception as exc:atomic_json(ROOT/'status.json',{'stage':'failed','error':repr(exc)});raise
