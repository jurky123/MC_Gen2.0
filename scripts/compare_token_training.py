"""Compare the actual pooled source and trained token model under their own protocols."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
from mcgen.disk_text import load_text_cache
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from train_five_task_reference import draw_sheet,fixed_noise
from validate_resolution import image_metrics

ROOT=Path('outputs/token_full_hq_protocol_review_20261109')
def read(p):return json.loads(p.read_text())

@torch.inference_mode()
def run():
    protocol=read(ROOT/'protocol.json');old=Path(protocol['old_evaluation']);new=Path(protocol['new_evaluation'])
    state=load_checkpoint(protocol['new_model']['checkpoint']);cfg=Experiment.from_dict(state['config'])
    encoder=load_text_cache(Path('outputs/token_full_hq_20261109/text_features'),state['text_identity'])
    model=PixelFlow(cfg.model).to('cuda:0').eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    rows=read(old/'broad/cases.json');(ROOT/'native').mkdir(exist_ok=True);(ROOT/'panels').mkdir(exist_ok=True)
    h,m=encoder.encode(conditioned_prompts([r['prompt'] for r in rows],cfg),'cuda:0')
    metrics=[]
    for si,seed in enumerate(protocol['seeds']):
        for start in range(0,len(rows),16):
            batch=rows[start:start+16]
            values=sample(model,h[start:start+16],m[start:start+16],seeds=[seed]*len(batch),steps=32,
                text_scale=2.,initial_noise=fixed_noise(len(batch),seed,'cuda:0'))
            assert torch.isfinite(values).all()
            for row,value in zip(batch,values):
                path=ROOT/'native'/f'{row["case_id"]}_s{si}.png';Image.fromarray(decode(value)).save(path)
                truth=np.array(Image.open(old/'broad/inputs'/f'{row["case_id"]}.png').convert('RGBA')).repeat(2,0).repeat(2,1)
                metrics.append({'split':row['group'],'mode':'text','variant':'tokens12000','case_id':row['case_id'],'seed':si,**image_metrics(path,truth)})
    old_metrics=read(old/'metrics.json')
    summaries=[r for r in old_metrics['summary'] if r['variant']=='text_best']
    new_metrics=read(new/'metrics.json')['summary']
    summaries += [dict(r,variant='tokens12000') for r in new_metrics if r['variant']=='last']
    for split in ('broad_val','replay_train'):
        part=[r for r in metrics if r['split']==split]
        summaries.append({'split':split,'mode':'text','variant':'tokens12000','outputs':len(part),
            **{k:float(np.mean([r[k] for r in part])) for k in ('premultiplied_mae','alpha_iou')}})
    cases=read(new/'samples.json');lookup={r['id']:r for r in read(old/'samples.json')}
    for split in ('test','val','train'):
        group=[r for r in cases if r['split']==split]
        for mode in ('joint_detail','text_detail','reference'):
            for start in range(0,len(group),8):
                entries=[]
                for row in group[start:start+8]:
                    prior=lookup[row['id']]
                    assert all(row[k]==prior[k] for k in ('target_hash','reference_sha256','captions'))
                    paths=[new/'inputs'/f'{row["case_id"]}_reference.png',new/'inputs'/f'{row["case_id"]}_target.png',
                        old/'native'/f'{prior["case_id"]}_text_best_{mode}_s0.png',new/'native'/f'{row["case_id"]}_last_{mode}_s0.png']
                    entries.append((paths,row['case_id']+' | '+row['captions'][0],'Matched targets, captions, noise | each model uses its own text encoder'))
                draw_sheet(ROOT/'panels'/f'{split}_{mode}_{start//8:02d}.png',split.upper()+' | '+mode,
                    ['HD reference','MC target','Pooled source4000','Token trained12000'],entries)
    for split in ('broad_val','replay_train'):
        group=[r for r in rows if r['group']==split]
        for start in range(0,len(group),8):
            entries=[]
            for row in group[start:start+8]:
                cid=row['case_id'];paths=[old/'broad/inputs'/f'{cid}.png',old/'broad/native'/f'{cid}_text_best_s0.png',ROOT/'native'/f'{cid}_s0.png']
                entries.append((paths,row['prompt'],split+' | fixed seed | pure T2I CFG2'))
            draw_sheet(ROOT/'panels'/f'{split}_{start//8:02d}.png',split+' | matched old evaluation targets',
                ['MC target','Pooled source4000','Token trained12000'],entries)
    for label in ('old_model','new_model'):
        assert file_hash(protocol[label]['checkpoint'])==protocol[label]['file_sha256']
    atomic_json(ROOT/'metrics.json',{'summary':summaries,'new_broad_per_output':metrics,
        'warning':'Alignment proxies, not semantic/edit correctness. Training data, budget and weighting also changed.'})
    body='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多token全量训练对比</title><style>body{font:16px/1.7 system-ui;background:#111827;color:#e5e7eb;max-width:1120px;margin:24px auto;padding:16px}img{max-width:100%}a{color:#93c5fd}details{padding:16px;background:#1f2937;margin:16px 0}</style><h1>旧汇总向量与多token全量训练</h1><p>旧模型使用正确的汇总向量，新模型使用有效token隐藏状态。旧模型为本轮实际初始化来源4000步，新模型新增12000步；不把新token初始化0步当作旧模型效果。</p><p>参考测试56、额外验证32、训练24；广域验证48、此前回放训练48，双种子统计。图片展示固定种子A，保留失败样本。数据范围、预算与权重同时改变，不能将收益独立归因于token结构。</p><p><a href="/token-full-evaluation/">新协议初始化、最佳与终点的完整双种子评估</a></p>'
    for prefix,title in [('test_','参考测试'),('val_','参考额外验证'),('train_','参考训练'),('broad_val_','新广域文生图验证'),('replay_train_','此前回放训练样本')]:
        body+='<details'+(' open' if prefix=='test_' else '')+'><summary>'+title+'</summary>'
        for p in sorted((ROOT/'panels').glob(prefix+'*.png')):body+=f'<p>{p.name}</p><a href="panels/{p.name}"><img loading="lazy" src="panels/{p.name}"></a>'
        body+='</details>'
    (ROOT/'index.html').write_text(body)
    publish=Path('outputs/native16_data_review_20261107/token-comparison');publish.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'panels',publish/'panels',dirs_exist_ok=True);shutil.copy2(ROOT/'index.html',publish/'index.html')
    atomic_json(ROOT/'verification.json',{'complete':True,'new_broad_outputs':len(metrics),'reference_cases_matched':len(cases),'source_checkpoints_unchanged':True})
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);torch.cuda.set_device('cuda:0');torch.cuda.set_per_process_memory_fraction(.4,'cuda:0');run()
