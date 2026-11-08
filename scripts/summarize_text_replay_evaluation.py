"""Render training diagnostics and a concise, auditable replay comparison."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json
from mcgen.runtime import load_checkpoint
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.sampling import sample
from mcgen.images import decode
from mcgen.catalog import file_hash
from train_five_task_reference import draw_sheet
from train_five_task_reference import fixed_noise
from evaluate_text_replay import VARIANTS, LABELS
from validate_resolution import image_metrics


def read(path):
    return json.loads(path.read_text())


def diagnostics(root,out,plot_python='python'):
    panels=out/'panels';panels.mkdir(exist_ok=True,parents=True)
    history=[json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    train=[json.loads(line) for line in (root/'training/metrics.jsonl').read_text().splitlines()]
    subprocess.run([plot_python,'scripts/plot_text_replay_metrics.py','--root',str(root),
                    '--out',str(panels/'training_diagnostics.png')],check=True)
    pairs={r['id']:r for r in read(root/'pairs.json')}
    controls=read(root/'generation_checks/000000/text_control.json')
    for start in range(0,len(controls),4):
        entries=[]
        for j in range(start,min(start+4,len(controls))):
            row=controls[j];target=pairs[row['id']]
            paths=[target['reference_file'],target['target_file']]
            for step in (0,4000,8000):
                paths += [root/'generation_checks'/f'{step:06d}'/f'control_{j:02d}_{key}.png' for key in ('joint_a','joint_b')]
            entries.append((paths,'A: '+row['text_a'],'B: '+row['text_b']))
        draw_sheet(panels/f'text_response_{start//4:02d}.png',
            'Fixed reference, changed existing caption | BF16 monitor | response is not edit correctness',
            ['Fixed reference','MC target','Before text A','Before text B','Text best A','Text best B','Final text A','Final text B'],entries)
    last=train[-1000:]
    atomic_json(out/'training_health.json',{'updates':len(train),'all_monitor_time_checks_unchanged':all(r['frozen_time_mlp_bitwise_unchanged'] for r in history),
        'max_preclip_gradient':max(r['gradient_norm'] for r in train),
        'last1000_median_gradient':statistics.median(r['gradient_norm'] for r in last),
        'last1000_clipped_fraction':sum(r['gradient_norm']>1 for r in last)/len(last),
        'stop_reason':read(root/'status.json')['stop_reason'],
        'caption_response':[{'step':r['step'],**r['caption_response_l1']} for r in history]})
    print('DIAGNOSTIC PANELS READY',flush=True)


def premultiplied(path):
    a=np.array(Image.open(path).convert('RGBA'),dtype=np.float32)/255
    a[...,:3]*=a[...,3:]
    return a


@torch.inference_mode()
def foundation(out,device):
    meta=read(Path('docs/models/step4128.json'))
    checkpoint=Path(meta['inference_checkpoint'])
    if file_hash(checkpoint)!=meta['inference_sha256']:raise ValueError('Foundation checkpoint changed')
    state=load_checkpoint(checkpoint)
    if state['text_identity']!=read(out/'protocol.json')['models']['before']['text_identity']:
        raise ValueError('Foundation text identity differs')
    cfg=Experiment.from_dict(state['config'])
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.35,device)
    model=PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    del state
    broad=out/'broad';cases=read(broad/'cases.json')
    f=torch.load(broad/'features.pt',weights_only=True)
    f={k:v.to(device) for k,v in f.items()}
    dest=out/'foundation4128';(dest/'native').mkdir(parents=True,exist_ok=True)
    metrics=[]
    for si,seed in enumerate((2026110841,2026110842)):
        for start in range(0,len(cases),16):
            selected=cases[start:start+16]
            values=sample(model,f['text'][start:start+16],f['mask'][start:start+16],
                seeds=[seed]*len(selected),steps=32,text_scale=2.,
                initial_noise=fixed_noise(len(selected),seed,device))
            if not bool(torch.isfinite(values).all()):raise FloatingPointError('Foundation output nonfinite')
            for row,value in zip(selected,values):
                path=dest/'native'/f'{row["case_id"]}_s{si}.png'
                Image.fromarray(decode(value)).save(path)
                truth=np.array(Image.open(broad/'inputs'/f'{row["case_id"]}.png').convert('RGBA')).repeat(2,0).repeat(2,1)
                metrics.append({'case_id':row['case_id'],'split':row['group'],'seed':si,**image_metrics(path,truth)})
    summary=[{'split':group,'variant':'foundation4128','mode':'text','outputs':sum(r['split']==group for r in metrics),
        **{k:float(np.mean([r[k] for r in metrics if r['split']==group])) for k in ('premultiplied_mae','alpha_iou')}}
        for group in ('broad_val','replay_train')]
    atomic_json(dest/'metrics.json',{'summary':summary,'per_output':metrics,
        'checkpoint_sha256':file_hash(checkpoint),'source_unchanged':file_hash(checkpoint)==meta['inference_sha256'],
        'purpose':'Original4128 T2I baseline only; not a reference-capable comparator',
        'precision':'float32','steps':32,'text_scale':2.})
    print('FOUNDATION T2I COMPLETE',json.dumps(summary),flush=True)


def finish(root,out,publish):
    if not read(out/'verification.json')['complete']:raise ValueError('Full evaluation is not complete')
    metrics=read(out/'metrics.json')
    rows=read(out/'samples.json')
    for split in ('test','val','train'):
        entries=[]
        for row in [r for r in rows if r['split']==split][:4]:
            cid=row['case_id']
            paths=[out/'inputs'/f'{cid}_reference.png',out/'inputs'/f'{cid}_target.png']
            paths += [out/'native'/f'{cid}_{v}_joint_detail_s0.png' for v in VARIANTS]
            entries.append((paths,cid+' | '+row['captions'][0],'First4 fixed cases; no visual filtering | FP32 | detail + reference'))
        draw_sheet(out/'panels'/f'overview_{split}.png',split.upper()+' | reference generation | fixed seed',
                   ['HD reference','MC target']+LABELS,entries)
    entries=[]
    for row in [r for r in read(out/'broad/cases.json') if r['group']=='broad_val'][:4]:
        cid=row['case_id']
        paths=[out/'broad/inputs'/f'{cid}.png']+[out/'broad/native'/f'{cid}_{v}_s0.png' for v in VARIANTS]
        entries.append((paths,row['prompt'],'First4 fresh validation cases | pure text | CFG2 | FP32'))
    draw_sheet(out/'panels/overview_broad.png','Fresh broad T2I validation | fixed seed',['MC target']+LABELS,entries)
    sensitivity=[]
    for split in ('test','val','train'):
        group=[r for r in rows if r['split']==split]
        for variant in VARIANTS:
            joint_image=[];short_detail=[]
            for row in group:
                for seed in range(2):
                    cid=row['case_id']
                    joint=premultiplied(out/'native'/f'{cid}_{variant}_joint_detail_s{seed}.png')
                    image=premultiplied(out/'native'/f'{cid}_{variant}_reference_s{seed}.png')
                    joint_image.append(float(np.abs(joint-image).mean()))
                    if len(row['captions'])>1:
                        short=premultiplied(out/'native'/f'{cid}_{variant}_joint_short_s{seed}.png')
                        short_detail.append(float(np.abs(joint-short).mean()))
            sensitivity.append({'split':split,'variant':variant,'joint_vs_image_l1':float(np.mean(joint_image)),
                'joint_short_vs_detail_l1':float(np.mean(short_detail)) if short_detail else None,
                'identical_joint_and_image_fraction':float(np.mean(np.array(joint_image)==0)),
                'interpretation':'Output response only, not correctness of an instructed edit.'})
    frozen=[]
    baseline=None
    for name,meta in read(out/'protocol.json')['models'].items():
        state=load_checkpoint(meta['checkpoint'])
        time={k:v for k,v in state['model'].items() if k.startswith('time.')}
        if baseline is None:baseline={k:v.clone() for k,v in time.items()}
        equal=all(torch.equal(v,baseline[k]) for k,v in time.items())
        frozen.append({'variant':name,'time_mlp_exactly_equal_to_initializer':equal})
        del state;gc.collect()
    if not all(r['time_mlp_exactly_equal_to_initializer'] for r in frozen):raise ValueError('Frozen time MLP changed')
    summary={'benchmarks':metrics['summary'],'conditioning_response':sensitivity,
             'frozen_time_verification':frozen,'training_health':read(out/'training_health.json'),
             'selection_note':'No checkpoint selected or promoted using test results.'}
    foundation_metrics=out/'foundation4128/metrics.json'
    if foundation_metrics.exists():
        summary['original_t2i_foundation']=read(foundation_metrics)['summary']
        entries=[]
        for row in [r for r in read(out/'broad/cases.json') if r['group']=='broad_val'][:8]:
            cid=row['case_id']
            paths=[out/'broad/inputs'/f'{cid}.png',out/'foundation4128/native'/f'{cid}_s0.png']
            paths += [out/'broad/native'/f'{cid}_{v}_s0.png' for v in ('before','text_best','last')]
            entries.append((paths,row['prompt'],'Fresh broad validation | same noise and CFG2 | FP32'))
        draw_sheet(out/'panels/foundation_text_comparison.png','Original T2I foundation versus reference-trained versions',
            ['MC target','Original T2I4128','Before (old8000)','Text best (+4000)','Final (+8000)'],entries)
    atomic_json(out/'review_summary.json',summary)
    path=out/'index.html';body=path.read_text()
    if 'id="additional-review"' not in body:
        extra='<section id="additional-review"><h2>训练诊断与文字响应</h2><p>响应幅度不等于编辑成功；这里换的是已有完整描述，并无冲突编辑真值。</p>'
        for name in ('training_diagnostics.png','text_response_00.png','text_response_01.png'):
            extra+=f'<a href="panels/{name}"><img loading="lazy" src="panels/{name}"></a>'
        if foundation_metrics.exists():
            extra+='<h2>与原始文生图底座比较</h2><p>该4128步模型仅比较纯文生图，不用于参考图任务。</p><img loading="lazy" src="panels/foundation_text_comparison.png">'
        extra+='</section>'
        body+=extra;path.write_text(body)
    shutil.copytree(out/'panels',publish/'panels',dirs_exist_ok=True)
    shutil.copy2(path,publish/'index.html')
    print('REVIEW SUMMARY COMPLETE',flush=True)
    for r in metrics['summary']:
        if (r['split']=='test' and r['mode']=='joint_detail') or r['split'] in ('broad_val','replay_train'):
            print(json.dumps(r),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('diagnostics','finish','foundation'))
    p.add_argument('--root',type=Path,default=Path('outputs/reference_text_replay2000_20261108_balanced'))
    p.add_argument('--out',type=Path,default=Path('outputs/reference_text_replay2000_review_20261108'))
    p.add_argument('--plot-python',default='python',help='Python environment with matplotlib; can differ from the training environment')
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.stage=='diagnostics':diagnostics(a.root,a.out,a.plot_python)
    elif a.stage=='foundation':foundation(a.out,a.device)
    else:finish(a.root,a.out,Path('outputs/native16_data_review_20261107/replay-evaluation'))
