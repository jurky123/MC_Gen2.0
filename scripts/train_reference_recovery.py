"""Long-budget native16 reference learning, then cautious joint fine-tuning."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import signal
import statistics
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from mcgen.catalog import atomic_json,file_hash
from mcgen.curated_training import initialize_mixed
from mcgen.images import decode
from mcgen.reference_adapter import backbone_digest
from mcgen.reference_recovery import RecoveryData,learning_rates,phase_for,trainable_phase
from mcgen.runtime import load_checkpoint,save_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from prepare_reference_recovery import COMMON
from train_curated_mixed import loss_for
from train_visual_reference import thumbnail


def initialize(data,device):
    info=json.loads((data.root/'initialization.json').read_text())
    if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Initialization changed')
    state=load_checkpoint(info['checkpoint'])
    if state.get('diagnostic_only') or state.get('target_leakage'):raise ValueError('Diagnostic weights are not production initializers')
    if data.cfg.text.backend=='qwen3_vl_tokens':
        from mcgen.config import Experiment
        from mcgen.model import PixelFlow
        transition=info.get('text_protocol_transition',{})
        if (transition.get('source_identity')!=state['text_identity']
                or transition.get('target_identity')!=data.encoder.identity
                or state['visual_identity']!=data.visual_identity):
            raise ValueError('Unverified pooled-to-token initialization')
        source=Experiment.from_dict(state['config'])
        a,b=source.to_dict()['model'],data.cfg.to_dict()['model']
        a.pop('text_length');b.pop('text_length')
        if a!=b or source.text.backend!='qwen3_vl_embedding':
            raise ValueError('Token initialization may change only text sequence length and protocol')
        torch.manual_seed(data.cfg.train.seed)
        model=PixelFlow(data.cfg.model).to(device)
        model.load_state_dict(state['model'],strict=True)
        return model,info
    if state['text_identity']!=data.encoder.identity or state['visual_identity']!=data.visual_identity:
        raise ValueError('Frozen encoder identity mismatch')
    torch.manual_seed(data.cfg.train.seed)
    return initialize_mixed(data.cfg,state,device),info


def optimizer(model,recipe):
    def text_parameter(name):
        return not name.startswith('visual_adapter.') and (name.startswith('text.') or '.cross_attention.' in name)
    base=[p for n,p in model.named_parameters() if not n.startswith('visual_adapter.')
          and not ('text_lr' in recipe and text_parameter(n))]
    groups=[{'params':base,'lr':recipe['base_lr']},
            {'params':model.visual_adapter.parameters(),'lr':recipe['visual_lr']}]
    if 'text_lr' in recipe:
        groups.append({'params':[p for n,p in model.named_parameters() if text_parameter(n)],'lr':recipe['text_lr']})
    return torch.optim.AdamW(groups,weight_decay=.01)


def check(root,device):
    data=RecoveryData(root);data.cache_device(device)
    recipe=json.loads((root/'recipe.json').read_text())
    model,info=initialize(data,device);opt=optimizer(model,recipe)
    before=backbone_digest(model)
    visual=model.visual_adapter.projection[1].weight.detach().clone()
    trainable_phase(model,'visual')
    ids,tasks,variants=data.samplers['visual'].sample(128,torch.Generator().manual_seed(1901))
    opt.zero_grad(set_to_none=True)
    loss_for(model,data.batch(ids,tasks,variants,device),device,1931)[0].backward()
    visual_grad=sum(p.grad.detach().float().square().sum().item() for p in model.visual_adapter.parameters() if p.grad is not None)**.5
    opt.step()
    frozen_exact=before==backbone_digest(model)
    visual_changed=not torch.equal(visual,model.visual_adapter.projection[1].weight)
    trainable_phase(model,'joint')
    hq=[i for i,r in enumerate(data.rows) if r['split']=='train' and r['quality_tier']=='highest_quality'][:16]
    text_before=model.text.weight.detach().clone()
    opt.zero_grad(set_to_none=True)
    loss_for(model,data.batch(hq,['text']*len(hq),[1]*len(hq),device),device,1973)[0].backward()
    no_visual_gradient=all(p.grad is None for p in model.visual_adapter.parameters())
    opt.step()
    text_changed=not torch.equal(text_before,model.text.weight)
    measurements=[]
    for phase in ('visual','joint'):
        trainable_phase(model,phase)
        times=[]
        torch.cuda.reset_peak_memory_stats(device)
        for k in range(5):
            ids,tasks,variants=data.samplers[phase].sample(128,torch.Generator().manual_seed(2003+k))
            batch=data.batch(ids,tasks,variants,device)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device);started=time.monotonic()
            loss_for(model,batch,device,2027+k)[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize(device);times.append(time.monotonic()-started)
        seconds=statistics.median(times[1:])
        measurements.append({'phase':phase,'batch_size':128,'seconds_per_step':seconds,
            'images_per_second':128/seconds,'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3})
    report={'passed':frozen_exact and visual_changed and visual_grad>0 and text_changed and no_visual_gradient,
        'visual_only_step_keeps_backbone_bitwise_unchanged':frozen_exact,'visual_weights_updated':visual_changed,
        'visual_gradient_norm':visual_grad,'text_task_updates_shared_generator':text_changed,
        'text_task_has_no_visual_parameter_gradient':no_visual_gradient,
        'measurements':measurements,'benchmark_weights_discarded':True,'fixed_updates':8000,
        'initial_checkpoint_unchanged':file_hash(info['checkpoint'])==info['sha256']}
    atomic_json(root/'checks.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise RuntimeError('Recovery training checks failed')


def select_validation(data):
    rng=torch.Generator().manual_seed(2026110817)
    result=[];seen=set()
    for quality in ('highest_quality','evaluation_only'):
        groups=defaultdict(list);unique={}
        for i,row in enumerate(data.rows):
            if row['split']=='val' and row['quality_tier']==quality:
                unique.setdefault(row['target_hash'],i)
        for i in unique.values():groups[data.categories[i]].append(i)
        for c,pool in groups.items():
            order=torch.randperm(len(pool),generator=rng).tolist();groups[c]=[pool[j] for j in order]
        chosen=[]
        while len(chosen)<24:
            for c in sorted(groups):
                if groups[c] and len(chosen)<24:
                    i=groups[c].pop()
                    if data.rows[i]['target_hash'] not in seen:
                        chosen.append(i);seen.add(data.rows[i]['target_hash'])
            if not any(groups.values()) and len(chosen)<24:raise ValueError('Insufficient independent validation targets')
        result+=chosen
    return result


def selection(data):
    valid=select_validation(data)
    previous=json.loads(Path('outputs/reference_overfit_20261108/samples.json').read_text())
    lookup={row['id']:i for i,row in enumerate(data.rows)}
    training=[lookup[r['id']] for r in previous]
    if len(training)!=16 or any(data.rows[i]['split']!='train' for i in training):raise ValueError('Training monitor selection changed')
    groups=defaultdict(list)
    for j,i in enumerate(valid):groups[data.categories[i]].append(j)
    wrong=[];same=[]
    for j,i in enumerate(valid):
        pool=groups[data.categories[i]]
        other=pool[(pool.index(j)+1)%len(pool)] if len(pool)>1 else (j+1)%len(valid)
        wrong.append(other);same.append(data.categories[valid[other]]==data.categories[i])
    return valid,training,wrong,same


@torch.inference_mode()
def evaluate(model,data,root,step,device,valid,monitors,wrong,same):
    directory=root/'generation_checks'/f'{step:06d}';directory.mkdir(parents=True,exist_ok=True)
    scores={'step':step,'native_size':16,'flow':{},'reference_permutation_same_category_count':sum(same)}
    for mode,variant in [('joint_coarse',2),('reference',0),('text',1)]:
        task='joint' if mode=='joint_coarse' else mode
        target,text,mask,refs,present=data.batch(valid,[task]*len(valid),[variant]*len(valid),device)
        for k,level in enumerate((.1,.5,.9,.99)):
            rng=torch.Generator(device=device).manual_seed(20261108+31*k)
            noise=torch.randn(target.shape,generator=rng,device=device)
            x=target*(1-level)+noise*level
            controls=[('matched',refs)] if mode=='text' else [('matched',refs),('wrong_reference',refs[wrong]),('no_reference',None)]
            values={}
            for label,reference in controls:
                parts=[]
                for start in range(0,len(valid),16):
                    stop=min(start+16,len(valid))
                    r=None if reference is None else reference[start:stop]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        ctx=model.context(text[start:stop],mask[start:stop],r,present[start:stop] if r is not None else None)
                        prediction=model(x[start:stop],torch.full((stop-start,),level,device=device),ctx)
                    parts.extend((prediction.float()-(noise-target)[start:stop]).square().mean((1,2,3)).cpu().tolist())
                values[label]=float(np.mean(parts))
                values[label+'_hq']=float(np.mean(parts[:24]));values[label+'_legacy']=float(np.mean(parts[24:]))
                if mode!='text' and any(same):values[label+'_same_class_subset']=float(np.mean([v for v,s in zip(parts,same) if s]))
            if mode!='text':values['reference_gain']=values['wrong_reference']-values['matched']
            scores['flow'][f'{mode}_t{level}']=values
    scores['reference_high_noise_mse']=float(np.mean([scores['flow'][f'{m}_t{t}']['matched'] for m in ('joint_coarse','reference') for t in (.9,.99)]))
    scores['joint_flow_mse']=float(np.mean([scores['flow'][f'joint_coarse_t{t}']['matched'] for t in (.1,.5,.9,.99)]))
    scores['text_flow_mse']=float(np.mean([scores['flow'][f'text_t{t}']['matched'] for t in (.1,.5,.9,.99)]))
    # Fixed train and validation targets; no test-set sampling or output filtering.
    panel=monitors+valid[:8]+valid[24:32]
    generated={}
    for label,task,variant,cfg in [('joint_coarse','joint',2,1.),('reference','reference',0,1.),
                                    ('text','text',1,1.),('joint_cfg2','joint',2,2.)]:
        blocks=[]
        for start in range(0,len(panel),16):
            indices=panel[start:start+16]
            _,h,m,r,p=data.batch(indices,[task]*len(indices),[variant]*len(indices),device)
            noise=torch.randn((1,4,16,16),generator=torch.Generator().manual_seed(2026110819)).repeat(len(indices),1,1,1)
            out=sample(model,h,m,r if task!='text' else None,p if task!='text' else None,
                seeds=[2026110819]*len(indices),steps=24,text_scale=cfg,ref_scale=cfg,initial_noise=noise)
            if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite validation generation')
            blocks.append(out.cpu())
        generated[label]=torch.cat(blocks)
        for j,value in enumerate(generated[label]):Image.fromarray(decode(value)).save(directory/f'{j:02d}_{label}.png')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    sheets=[]
    for page,start in enumerate(range(0,len(panel),8),1):
        group=panel[start:start+8]
        sheet=Image.new('RGB',(938,66+164*len(group)),'#111827');draw=ImageDraw.Draw(sheet)
        draw.text((14,10),f'Update{step} | current full dataset | same seed | native16',font=font,fill='white')
        for c,label in enumerate(('HD reference','MC target','Class + ref CFG1','Image only CFG1','Detailed text','Class + ref CFG2')):
            draw.text((14+c*154,34),label,font=font,fill='#cbd5e1')
        for local,i in enumerate(group):
            j=start+local;y=62+local*164;row=data.rows[i]
            with Image.open(row['reference_file']) as im:ref=thumbnail(im,128)
            pictures=[ref,thumbnail(Image.fromarray(decode(data.targets[i])),128,native=True)]
            pictures += [thumbnail(Image.fromarray(decode(generated[label][j])),128,native=True)
                         for label in ('joint_coarse','reference','text','joint_cfg2')]
            for c,picture in enumerate(pictures):sheet.paste(picture,(14+c*154,y))
            draw.text((14,y+132),f'{row["split"]} | {row["captions"][0]}'[:108],font=font,fill='white')
            draw.text((14,y+147),'Reference task text: '+data.categories[i],font=font,fill='#aebbd0')
        name=f'comparison_{page:02d}.png';sheet.save(directory/name);sheets.append(name)
    atomic_json(directory/'panel.json',[{**data.rows[i],'coarse_condition':data.categories[i]} for i in panel])
    h,m=data.encoder.encode(conditioned_prompts(COMMON,data.cfg),device)
    noise=torch.randn((1,4,16,16),generator=torch.Generator().manual_seed(2026110823)).repeat(len(COMMON),1,1,1)
    out=sample(model,h,m,seeds=[2026110823]*len(COMMON),steps=24,text_scale=2.,initial_noise=noise)
    common=torch.stack([torch.from_numpy(decode(x)) for x in out.cpu()])
    baseline=root/'text_baseline.pt'
    if step==0:torch.save(common,baseline)
    old=torch.load(baseline,weights_only=True)
    scores['common_text_mean_uint8_drift']=(common.float()-old.float()).abs().mean().item()
    scores['common_text_max_uint8_drift']=(common.to(torch.int16)-old.to(torch.int16)).abs().max().item()
    common_sheet=Image.new('RGB',(334,66+164*len(COMMON)),'#111827');draw=ImageDraw.Draw(common_sheet)
    draw.text((14,10),'Text-only regression | fixed prompts and Gaussian noise',font=font,fill='white')
    draw.text((14,34),'Initial step270',font=font,fill='#cbd5e1')
    draw.text((174,34),f'Current update{step}',font=font,fill='#cbd5e1')
    for j,prompt in enumerate(COMMON):
        y=62+j*164
        for c,pixels in enumerate((old[j],common[j])):
            common_sheet.paste(thumbnail(Image.fromarray(pixels.numpy()),128,native=True),(14+c*160,y))
        draw.text((14,y+132),prompt,font=font,fill='white')
        Image.fromarray(common[j].numpy()).save(directory/f'common_{j:02d}.png')
    common_sheet.save(directory/'text_regression.png')
    atomic_json(directory/'metrics.json',scores)
    atomic_json(root/'validation_latest.json',scores)
    return scores


def train(root,device):
    data=RecoveryData(root);data.cache_device(device)
    cfg=data.cfg;recipe=json.loads((root/'recipe.json').read_text());contract=file_hash(root/'recipe.json')
    checks=json.loads((root/'checks.json').read_text())
    if not checks['passed']:raise ValueError('Checks did not pass')
    model,initialization=initialize(data,device);opt=optimizer(model,recipe)
    out=root/'training';out.mkdir(exist_ok=True)
    valid,monitors,wrong,same=selection(data)
    step=0;best_ref=best_text=best_joint=float('inf');counts=Counter();conditions=Counter()
    validation_seconds=0.;validation_calls=0;training_seconds=0.;frozen_verified=False
    if (out/'latest.pt').exists():
        state=load_checkpoint(out/'latest.pt')
        if state['recipe_sha256']!=contract or state['catalog_fingerprint']!=data.pairs_sha256 or state['config']!=cfg.to_dict():
            raise ValueError('Resume contract changed')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        step=state['step'];best_ref=state['best_reference'];best_text=state['best_text'];best_joint=state['best_joint']
        counts=Counter(state['task_counts']);conditions=Counter(state['condition_counts'])
        frozen_verified=state.get('visual_stage_frozen_verified',False)
    initial_backbone=initialization['initial_backbone_sha256']
    atomic_json(root/'run.json',{'config':cfg.to_dict(),'recipe':recipe,'recipe_sha256':contract,
        'initialization':initialization,'pairs_sha256':data.pairs_sha256,'validation_indices':valid,
        'wrong_reference_local_indices':wrong,'wrong_reference_same_category':same,'training_monitor_indices':monitors,
        'test_used_for_selection':False,'native_size':16,'cached_visual_tokens_on_gpu':True})
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    def save(name):
        save_checkpoint(out/name,{'config':cfg.to_dict(),'model':model.state_dict(),'optimizer':opt.state_dict(),
            'step':step,'best_reference':best_ref,'best_text':best_text,'best_joint':best_joint,
            'task_counts':dict(counts),'condition_counts':dict(conditions),'recipe_sha256':contract,
            'catalog_fingerprint':data.pairs_sha256,'text_identity':data.encoder.identity,
            'visual_identity':data.visual_identity,'initialization':initialization,
            'visual_stage_frozen_verified':frozen_verified,
            'training_protocol':'reference_recovery_native16_v1','phase':phase_for(min(step,recipe['steps']-1),recipe)})
    def validate():
        nonlocal best_ref,best_joint,best_text,validation_seconds,validation_calls,frozen_verified
        started=time.monotonic()
        scores=evaluate(model,data,root,step,device,valid,monitors,wrong,same)
        if step<=recipe['visual_warmup_steps']:
            exact=backbone_digest(model)==initial_backbone
            scores['backbone_frozen_exact']=exact
            if not exact:raise RuntimeError('Backbone changed during visual-only warmup')
            if step==recipe['visual_warmup_steps']:
                frozen_verified=True
                atomic_json(root/'visual_stage_regression.json',{'passed':True,'step':step,
                    'backbone_sha256':initial_backbone,'common_text_max_uint8_drift':scores['common_text_max_uint8_drift']})
        if scores['reference_high_noise_mse']<best_ref:
            best_ref=scores['reference_high_noise_mse'];save('best_reference.pt')
        if scores['joint_flow_mse']<best_joint:
            best_joint=scores['joint_flow_mse'];save('best_joint_flow.pt')
        if scores['text_flow_mse']<best_text:
            best_text=scores['text_flow_mse'];save('best_text_flow.pt')
        save('latest.pt')
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(scores)+'\n')
        validation_seconds+=time.monotonic()-started;validation_calls+=1
        atomic_json(root/'validation_latest.json',scores)
        atomic_json(root/'generation_checks'/f'{step:06d}'/'metrics.json',scores)
        publish_review(root,step)
        print('VALIDATION',step,'reference',scores['reference_high_noise_mse'],'text',scores['text_flow_mse'],flush=True)
    started=time.monotonic()
    current_phase=None
    if step==0:
        trainable_phase(model,'visual');validate()
    timings={r['phase']:r['seconds_per_step'] for r in checks['measurements']}
    for update in range(step,recipe['steps']):
        if stopping[0]:break
        begin=time.monotonic();phase=phase_for(update,recipe)
        if phase!=current_phase:
            if phase=='joint' and not frozen_verified:raise RuntimeError('Visual-stage regression check missing')
            trainable_phase(model,phase);current_phase=phase
            print('PHASE',phase,'at',update,flush=True)
        rng=torch.Generator().manual_seed(cfg.train.seed+1000003*update)
        ids,tasks,variants=data.samplers[phase].sample(recipe['batch_size'],rng)
        batch=data.batch(ids,tasks,variants,device)
        rates=learning_rates(update,recipe)
        for group,lr in zip(opt.param_groups,rates):group['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss,parts=loss_for(model,batch,device,cfg.train.seed+update+17)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        step=update+1;counts.update(tasks)
        conditions.update('joint_coarse' if t=='joint' and v==2 else 'joint_short' if t=='joint' else t for t,v in zip(tasks,variants))
        training_seconds+=time.monotonic()-begin
        metric={'step':step,'phase':phase,'loss':loss.item(),'gradient_norm':norm.item(),
            'base_lr':rates[0],'visual_lr':rates[1],'elapsed_seconds':time.monotonic()-started,
            'task_counts':dict(counts),'condition_counts':dict(conditions)}
        for task in set(tasks):metric['loss_'+task]=parts.detach()[torch.tensor([j for j,t in enumerate(tasks) if t==task],device=device)].mean().item()
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        remaining_visual=max(0,recipe['visual_warmup_steps']-step)
        remaining_joint=recipe['steps']-max(step,recipe['visual_warmup_steps'])
        remaining_checks=sum(s>step for s in recipe['evaluate_updates'])
        eta=remaining_visual*timings['visual']+remaining_joint*timings['joint']
        eta+=remaining_checks*validation_seconds/max(1,validation_calls)
        atomic_json(root/'status.json',{**metric,'complete':False,'planned_steps':recipe['steps'],
            'estimated_remaining_seconds':eta,'training_seconds':training_seconds,
            'average_draws_per_train_target':step*recipe['batch_size']/864})
        if step%100==0:print('STEP',step,phase,round(loss.item(),6),flush=True)
        if step in recipe['evaluate_updates']:validate()
        elif step%500==0:save('latest.pt')
        if step==recipe['visual_warmup_steps']:save('visual_warmup.pt')
    save('latest.pt')
    atomic_json(root/'status.json',{'step':step,'planned_steps':recipe['steps'],'complete':step==recipe['steps'],
        'stopped':stopping[0],'seconds':time.monotonic()-started,'best_reference':best_ref,'best_joint':best_joint,
        'best_text':best_text,'task_counts':dict(counts),'condition_counts':dict(conditions),
        'initial_checkpoint_unchanged':file_hash(initialization['checkpoint'])==initialization['sha256']})


def publish_review(root,latest):
    import shutil
    destination=Path('outputs/native16_data_review_20261107/recovery')
    (destination/'panels').mkdir(parents=True,exist_ok=True)
    records=[]
    for folder in sorted((root/'generation_checks').iterdir(),reverse=True):
        if not (folder/'metrics.json').exists():continue
        score=json.loads((folder/'metrics.json').read_text())
        images=[]
        for path in sorted(folder.glob('comparison_*.png'))+[folder/'text_regression.png']:
            if not path.is_file():continue
            filename=f'{folder.name}_{path.name}'
            shutil.copy2(path,destination/'panels'/filename)
            images.append(filename)
        records.append((score,images))
    body='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>参考图高预算训练</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:1050px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%}details{background:#1f2937;padding:16px;margin:18px 0;border-radius:10px}summary{cursor:pointer}h3{font-size:16px}</style><p><a href="/">数据对照</a> · <a href="/overfit/">小样本拟合</a> · <a href="/evaluation/">上一轮测评</a></p><h1>完整无损16格数据 · 高预算参考图训练</h1><p>同一批1064对训练配对/864个目标。前2000步只训练视觉分支，后6000步小学习率联合训练；类别条件让参考图提供设计，文生图保留完整描述。每次验证后更新，刷新查看。</p><p>对照列：高清参考、MC目标、类别词+参考CFG1、纯参考CFG1、精标文生图、类别词+参考CFG2。前两张对照为训练样本，后两张为验证样本；不使用测试集选模型。文字回归图左为本轮初始270步模型，右为当前权重。</p>'''
    for score,images in records:
        step=score['step'];phase='视觉分支阶段' if step<=2000 else '联合微调阶段'
        body+=f'<details {"open" if step==latest else ""}><summary>更新{step}/8000 · {phase} · 参考高噪声误差 {score["reference_high_noise_mse"]:.5f} · 文本误差 {score["text_flow_mse"]:.5f}</summary>'
        if 'backbone_frozen_exact' in score:body+='<p>底座权重逐元素保持不变：'+str(score['backbone_frozen_exact'])+'</p>'
        for filename in images:body+=f'<p><a href="panels/{filename}">{filename}</a></p><img loading="lazy" src="panels/{filename}">'
        body+='</details>'
    path=destination/'index.html';tmp=path.with_suffix('.html.tmp');tmp.write_text(body);tmp.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','train'])
    parser.add_argument('--out',type=Path,default=Path('outputs/reference_recovery_20261108'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.cuda.set_per_process_memory_fraction(.50,args.device)
    if args.stage=='check':check(args.out,args.device)
    else:train(args.out,args.device)
