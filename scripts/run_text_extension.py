"""Continue the expansion, then run its unchanged paired evaluation."""
import os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json
ROOT=Path('outputs/text_expansion22k_20261102')
if __name__=='__main__':
 if ROOT.exists():raise ValueError('Use a new output; existing continuation is preserved')
 env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
 stages=[('training',['scripts/continue_text_expansion.py']),('evaluation',['scripts/evaluate_text_recovery.py','--root',str(ROOT),'--device','cuda:1']),('summary',['scripts/summarize_text_recovery.py','--root',str(ROOT)]),('export',['scripts/export_text_expansion_evaluation.py','--root',str(ROOT)])]
 try:
  for stage,args in stages:
   if ROOT.exists():atomic_json(ROOT/'status.json',{'stage':stage,'updated_unix':time.time()})
   path=Path(str(ROOT)+'.training.log') if stage=='training' else ROOT/(stage+'.log')
   with path.open('a') as log:subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  atomic_json(ROOT/'status.json',{'stage':'complete_pending_visual_review','step':22000,'production_promoted':False})
 except Exception as exc:
  atomic_json(ROOT/'status.json',{'stage':'failed','error':repr(exc)});raise
