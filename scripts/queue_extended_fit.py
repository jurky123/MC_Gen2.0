"""Run frozen training-fit probes after the long training/evaluation has released its GPU."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('outputs/text_expansion22k_20261102')
if __name__=='__main__':
 while True:
  status=json.loads((ROOT/'status.json').read_text())
  if status['stage']=='failed':raise SystemExit('Main continuation failed; fit evaluation not run')
  if status['stage']=='complete_pending_visual_review':break
  time.sleep(30)
 with (ROOT/'fit_diagnosis.log').open('a') as log:
  subprocess.run([sys.executable,'scripts/diagnose_expansion_fit.py','--extended-root',str(ROOT)],env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
