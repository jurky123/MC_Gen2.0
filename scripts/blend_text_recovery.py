"""One predeclared 50/50 parameter interpolation; inference-only artifact."""
from pathlib import Path
import torch
from mcgen.catalog import file_hash,atomic_json
from mcgen.runtime import load_checkpoint,save_checkpoint
from mcgen.config import Experiment

if __name__=='__main__':
 root=Path('outputs/text_recovery_core_20261102');before=Path('outputs/text_caption_scale1152_20261102/training/mixed/latest.pt');after=root/'training/latest.pt';out=root/'balanced_candidate.pt'
 if out.exists():raise ValueError('Preserve previous candidate')
 a,b=load_checkpoint(before),load_checkpoint(after)
 assert a['text_identity']==b['text_identity'] and a['config']['model']==b['config']['model']
 assert a['model'].keys()==b['model'].keys()
 model={}
 for k,x in a['model'].items():
  y=b['model'][k]
  assert x.shape==y.shape and x.dtype==y.dtype
  if x.is_floating_point():model[k]=torch.lerp(x,y,.5)
  else:assert torch.equal(x,y);model[k]=x.clone()
 provenance={'method':'parameter_interpolation','recovery_weight':.5,'parents':[{'path':str(p),'sha256':file_hash(p)} for p in [before,after]],'purpose':'Single fixed coefficient to check preservation of detailed-prompt ability; evaluated on development set, not a new training run.'}
 save_checkpoint(out,{'config':b['config'],'model':model,'text_identity':b['text_identity'],'catalog_fingerprint':b['catalog_fingerprint'],'step':0,'inference_only':True,'weight_interpolation':provenance})
 atomic_json(root/'balanced_candidate.json',provenance)
 print(out)
