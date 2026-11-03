"""Paired project bootstrap for the frozen recovery development probes."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from mcgen.catalog import atomic_json

ROOT=Path('outputs/text_recovery_core_20261102')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--balanced',action='store_true');parser.add_argument('--best-flow',action='store_true');parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.root
 p=ROOT/('evaluation_balanced' if args.balanced else ('evaluation_best_flow' if args.best_flow else 'evaluation'));rows=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()];groups=defaultdict(dict)
 for r in rows:
  arm,group=r['mode'].split(':',1);groups[group].setdefault(arm,{})[(r['id'],r['seed'])]=r
 result={}
 for group,arms in groups.items():
  assert arms['before'].keys()==arms['recovery'].keys()
  values={}
  for metric in next(iter(arms['before'].values()))['metrics']:
   projects=defaultdict(list);missing=0
   for key,b in arms['before'].items():
    before,after=b['metrics'][metric],arms['recovery'][key]['metrics'][metric]
    if before is None or after is None:missing+=1;continue
    projects[b['lineage']].append(after-before)
   if not projects:continue
   means=np.array([np.mean(v) for v in projects.values()]);rng=np.random.default_rng(20261112)
   ci=np.quantile(means[rng.integers(len(means),size=(10000,len(means)))].mean(1),[.025,.975])
   values[metric]={'recovery_minus_before_project_mean':float(means.mean()),'paired_project_bootstrap_95':ci.tolist(),'missing_pairs':missing,'groups':len(projects)}
  result[group]={'cases':len(arms['before']),'metrics':values}
 result['limitations']='Development-set paired project estimates. Core/broad overlap; train_fit is memorization. Novel_composition is a prompt-control probe with possible seen combinations, grouped by category, not held-out generalization. No training-seed uncertainty.'
 atomic_json(p/'paired_analysis.json',result)
 for group,arms in groups.items():
  print(group,len(arms['before']))
  for metric in ['composite_mae','alpha_iou','no_foreground','requested_color_fraction']:
   if metric not in next(iter(arms['before'].values()))['metrics']:continue
   print(metric,{arm:round(float(np.mean([v['metrics'][metric] for v in rs.values()])),4) for arm,rs in arms.items()})
