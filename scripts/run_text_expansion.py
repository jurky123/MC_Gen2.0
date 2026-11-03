"""Run the frozen expansion through training and paired generation evaluation."""
import json,os,subprocess,sys,time
from pathlib import Path
from mcgen.catalog import atomic_json,file_hash
ROOT=Path('outputs/text_expansion2k_20261102')
BASE=Path('outputs/text_recovery_core_20261102')
def status(stage,**extra):
 atomic_json(ROOT/'run_status.json',{'stage':stage,'updated_unix':time.time(),**extra});print(stage,flush=True)
def command(stage,args):
 status(stage)
 with (ROOT/(stage+'.log')).open('a') as log:subprocess.run([sys.executable,*args],env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
 if (ROOT/'training').exists():raise ValueError('Training output already exists; do not silently restart')
 try:
  command('build',['scripts/build_text_expansion.py'])
  command('caption_export',['scripts/export_text_expansion_captions.py'])
  command('training',['scripts/train_caption_arm.py','--config',str(ROOT/'config.yaml'),'--catalog',str(ROOT/'catalog'),'--out',str(ROOT/'training'),'--initialize',str(BASE/'training/latest.pt'),'--device','cuda:1'])
  command('evaluation',['scripts/evaluate_text_recovery.py','--root',str(ROOT),'--device','cuda:1'])
  command('summary',['scripts/summarize_text_recovery.py','--root',str(ROOT)])
  command('evaluation_export',['scripts/export_text_expansion_evaluation.py'])
  status('complete_pending_visual_evaluation',checkpoint=str(ROOT/'training/latest.pt'),checkpoint_sha256=file_hash(ROOT/'training/latest.pt'),production_promoted=False)
 except Exception as exc:
  status('failed',error=repr(exc));raise
if __name__=='__main__':main()
