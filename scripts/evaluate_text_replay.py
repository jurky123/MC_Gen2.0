"""Compare all distinct replay checkpoints, with fresh broad T2I and held-out pairs."""
import argparse
from collections import Counter, defaultdict
import gc
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import TASK_VIEWS
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.text_replay import TextReplayData
from evaluate_lossless_native16 import model_hash
import evaluate_five_task_reference as shared
from train_five_task_reference import draw_sheet, fixed_noise
from validate_resolution import image_metrics

VARIANTS = ('before', 'text_best', 'reference_best', 'last')
LABELS = ['Before (old8000)', 'Text best (+4000)', 'Ref best (+2000)', 'Final (+8000)']
SEEDS = (2026110841, 2026110842)


def read(path):
    return json.loads(path.read_text())


def prepare(root, base, out):
    if not read(root/'status.json')['complete']:
        raise ValueError('Training must be complete')
    p = read(base/'protocol.json')
    if file_hash(base/'features.pt') != p['features_sha256']:
        raise ValueError('Reference evaluation features changed')
    if out.exists():
        protocol = read(out/'protocol.json')
        if protocol['base_protocol_sha256'] != file_hash(base/'protocol.json'):
            raise ValueError('Base evaluation changed')
        return
    out.mkdir(parents=True)
    (out/'native').mkdir()
    (out/'panels').mkdir()
    (out/'inputs').symlink_to((base/'inputs').resolve(), target_is_directory=True)
    (out/'features.pt').symlink_to((base/'features.pt').resolve())
    shutil.copy2(base/'samples.json', out/'samples.json')
    paths = {'before': Path(p['models']['previous']['checkpoint']),
             'text_best': root/'training/best_text.pt',
             'reference_best': root/'training/best_reference.pt',
             'last': root/'training/latest.pt'}
    models = {}
    for name, path in paths.items():
        state = load_checkpoint(path)
        if state.get('diagnostic_only') or state.get('target_leakage'):
            raise ValueError('Invalid evaluation checkpoint')
        if any(state[k] != p['models']['previous'][k] for k in ('text_identity', 'visual_identity')):
            raise ValueError('Evaluation encoder identities differ')
        models[name] = {'checkpoint': str(path.resolve()), 'step': state['step'],
            'file_sha256': file_hash(path), 'model_sha256': model_hash(state['model']),
            'text_identity': state['text_identity'], 'visual_identity': state['visual_identity']}
        del state
        gc.collect()
    atomic_json(out/'protocol.json', {**p, 'models': models,
        'base_evaluation': str(base.resolve()), 'base_protocol_sha256': file_hash(base/'protocol.json'),
        'base_balanced_best_is_initializer': p['models']['best']['model_sha256'] == models['before']['model_sha256'],
        'variants': list(VARIANTS), 'labels': LABELS,
        'selection': 'Use preselected validation bests and fixed endpoint; do not select using test results',
        'code_sha256': {s: file_hash(s) for s in ('scripts/evaluate_text_replay.py',
            'scripts/evaluate_five_task_reference.py', 'src/mcgen/sampling.py', 'src/mcgen/model.py')}})
    print('PREPARED FOUR CHECKPOINTS', {k: v['step'] for k, v in models.items()}, flush=True)


def import_base(base, out, source, destination):
    progress = base/f'progress_{source}.json'
    if not progress.exists() or read(progress).get('phase') != 'complete':
        return False
    own, old = read(out/'protocol.json'), read(base/'protocol.json')
    if own['models'][destination]['model_sha256'] != old['models'][source]['model_sha256']:
        raise ValueError('Cached checkpoint differs')
    rows = read(out/'samples.json')
    expected = {f'{r["case_id"]}_{source}_{n}_s{s}.png' for r in rows for n, _, _ in TASK_VIEWS for s in range(2)}
    from prepare_reference_recovery import COMMON
    expected |= {f'common_{j:02d}_{source}_s{s}.png' for j in range(len(COMMON)) for s in range(2)}
    for name in expected:
        path = base/'native'/name
        with Image.open(path) as im:
            if im.size != (16, 16) or im.mode != 'RGBA':
                raise ValueError('Invalid cached output')
        shutil.copy2(path, out/'native'/name.replace('_'+source+'_', '_'+destination+'_'))
    atomic_json(out/f'progress_{destination}.json', {'phase': 'complete',
        'reused_from': str(base), 'source_variant': source, 'outputs': len(expected)})
    print('REUSED', source, 'as', destination, len(expected), flush=True)
    return True


def broad_cases(root, out):
    cfg = Experiment.load(root/'config.yaml')
    identity = read(out/'protocol.json')['models']['before']['text_identity']
    data = TextReplayData(root, cfg, identity)
    monitored = set(data.validation_indices(1024))
    groups = defaultdict(list)
    for i, row in enumerate(data.datasets['val'].records):
        if i not in monitored:
            groups[row.get('category_15', 'other')].append(i)
    rng = torch.Generator().manual_seed(2026110929)
    for c, pool in groups.items():
        groups[c] = [pool[j] for j in torch.randperm(len(pool), generator=rng).tolist()]
    val = []
    while len(val) < 48:
        for c in sorted(groups):
            if groups[c] and len(val) < 48:
                val.append(groups[c].pop())
        if not any(groups.values()) and len(val) < 48:
            raise ValueError('Insufficient fresh broad validation targets')
    train_groups = defaultdict(list)
    for i, row in enumerate(data.datasets['train'].records):
        train_groups[row['replay_type']].append(i)
    train = [pool[int(torch.randint(len(pool), (), generator=rng))] for _, pool in sorted(train_groups.items())]
    selections = [('broad_val', 'val', val), ('replay_train', 'train', train)]
    cases, targets, features, masks = [], [], [], []
    for label, split, ids in selections:
        batch = data.batch(ids, ['text']*len(ids), 'cpu', split=split)
        targets.append(batch[0]); features.append(batch[1]); masks.append(batch[2])
        for number, idx in enumerate(ids):
            row = data.datasets[split].records[idx]
            cases.append({k: row.get(k) for k in ('id', 'target_hash', 'prompt', 'category_15', 'replay_type')}
                         | {'split': split, 'group': label, 'case_id': f'{label}_{number:02d}'})
    path = out/'broad'
    path.mkdir(exist_ok=True)
    (path/'native').mkdir(exist_ok=True)
    (path/'inputs').mkdir(exist_ok=True)
    x, h, m = torch.cat(targets), torch.cat(features), torch.cat(masks)
    for row, target in zip(cases, x):
        Image.fromarray(decode(target)).save(path/'inputs'/f'{row["case_id"]}.png')
    if {r['target_hash'] for r in cases if r['split'] == 'train'} & {r['target_hash'] for r in cases if r['split'] == 'val'}:
        raise ValueError('Broad train/validation leakage')
    atomic_json(path/'cases.json', cases)
    torch.save({'text': h, 'mask': m}, path/'features.pt')
    atomic_json(path/'protocol.json', {'counts': dict(Counter(r['group'] for r in cases)),
        'fresh_validation_excludes_all_1024_training_monitors': not (set(val) & monitored),
        'train_selection': 'one fixed random target from each of the48 replay text strata',
        'features_sha256': file_hash(path/'features.pt'), 'seeds': SEEDS, 'steps': 32,
        'text_scale': 2., 'precision': 'float32', 'no_quality_filtering': True})


@torch.inference_mode()
def broad_generate(out, variant, device):
    path = out/'broad'
    cases = read(path/'cases.json')
    protocol = read(path/'protocol.json')
    if file_hash(path/'features.pt') != protocol['features_sha256']:
        raise ValueError('Broad features changed')
    f = torch.load(path/'features.pt', weights_only=True)
    f = {k: v.to(device) for k, v in f.items()}
    meta = read(out/'protocol.json')['models'][variant]
    if file_hash(meta['checkpoint']) != meta['file_sha256']:
        raise ValueError('Checkpoint changed')
    state = load_checkpoint(meta['checkpoint'])
    cfg = Experiment.from_dict(state['config'])
    model = PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    del state
    for si, seed in enumerate(SEEDS):
        for start in range(0, len(cases), 16):
            ids = [i for i in range(start, min(start+16, len(cases)))
                   if not (path/'native'/f'{cases[i]["case_id"]}_{variant}_s{si}.png').exists()]
            if not ids:
                continue
            values = sample(model, f['text'][ids], f['mask'][ids], seeds=[seed]*len(ids), steps=32,
                text_scale=2., initial_noise=fixed_noise(len(ids), seed, device))
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('Nonfinite broad generation')
            for i, value in zip(ids, values):
                Image.fromarray(decode(value)).save(path/'native'/f'{cases[i]["case_id"]}_{variant}_s{si}.png')
    atomic_json(path/f'progress_{variant}.json', {'phase': 'complete', 'outputs': len(cases)*2})
    print('BROAD COMPLETE', variant, flush=True)


def render(root, out, publish):
    rows, protocol = read(out/'samples.json'), read(out/'protocol.json')
    for name, meta in protocol['models'].items():
        if read(out/f'progress_{name}.json')['phase'] != 'complete' or file_hash(meta['checkpoint']) != meta['file_sha256']:
            raise ValueError('Incomplete comparison or modified source checkpoint')
    panels, metric_rows = [], []
    for split in ('test', 'val', 'train'):
        group = [r for r in rows if r['split'] == split]
        for name, _, _ in TASK_VIEWS:
            for seed in range(2):
                for start in range(0, len(group), 8):
                    entries = []
                    for row in group[start:start+8]:
                        cid = row['case_id']
                        target = out/'inputs'/f'{cid}_target.png'
                        truth = np.array(Image.open(target).convert('RGBA')).repeat(2, 0).repeat(2, 1)
                        paths = [out/'native'/f'{cid}_{v}_{name}_s{seed}.png' for v in VARIANTS]
                        for variant, path in zip(VARIANTS, paths):
                            metric_rows.append({'split': split, 'case_id': cid, 'variant': variant,
                                'mode': name, 'seed': seed, **image_metrics(path, truth)})
                        entries.append(([out/'inputs'/f'{cid}_reference.png', target]+paths,
                            cid+' | '+row['captions'][0], 'Same reference, text and noise | FP32 | native16'))
                    filename = f'{split}_{name}_s{seed}_{start//8:02d}.png'
                    draw_sheet(out/'panels'/filename, split.upper()+' | '+name+' | seed'+str(seed+1),
                               ['HD reference', 'MC target']+LABELS, entries)
                    panels.append(filename)
    broad = out/'broad'
    from prepare_reference_recovery import COMMON
    for seed in range(2):
        filename = f'common_s{seed}.png'
        entries = [([out/'native'/f'common_{i:02d}_{v}_s{seed}.png' for v in VARIANTS],
                    prompt, 'Common pure T2I | CFG2 | matched noise') for i, prompt in enumerate(COMMON)]
        draw_sheet(out/'panels'/filename, 'Common T2I regression | seed'+str(seed+1), LABELS, entries)
        panels.append(filename)
    for group in ('broad_val', 'replay_train'):
        cases = [r for r in read(broad/'cases.json') if r['group'] == group]
        for seed in range(2):
            for start in range(0, len(cases), 8):
                entries = []
                for row in cases[start:start+8]:
                    cid = row['case_id']
                    target = broad/'inputs'/f'{cid}.png'
                    truth = np.array(Image.open(target).convert('RGBA')).repeat(2, 0).repeat(2, 1)
                    paths = [broad/'native'/f'{cid}_{v}_s{seed}.png' for v in VARIANTS]
                    for variant, path in zip(VARIANTS, paths):
                        metric_rows.append({'split': group, 'case_id': cid, 'variant': variant,
                            'mode': 'text', 'seed': seed, **image_metrics(path, truth)})
                    entries.append(([target]+paths, row['prompt'], group+' | pure T2I | CFG2 | FP32'))
                filename = f'{group}_s{seed}_{start//8:02d}.png'
                draw_sheet(out/'panels'/filename, group+' | pure text | seed'+str(seed+1), ['MC target']+LABELS, entries)
                panels.append(filename)
    summary = []
    for split, mode, variant in sorted({(r['split'], r['mode'], r['variant']) for r in metric_rows}):
        group = [r for r in metric_rows if (r['split'], r['mode'], r['variant']) == (split, mode, variant)]
        summary.append({'split': split, 'mode': mode, 'variant': variant, 'outputs': len(group),
            **{k: float(np.mean([r[k] for r in group])) for k in ('premultiplied_mae', 'alpha_iou')}})
    history = [json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    atomic_json(out/'metrics.json', {'summary': summary, 'per_output': metric_rows,
        'checkpoint_labels': dict(zip(VARIANTS, LABELS)),
        'metric_note': 'Pixel alignment proxies, not semantic correctness; test results do not select checkpoints.'})
    atomic_json(out/'monitor_history.json', [{k: r.get(k) for k in ('step','balanced_generation_mae',
        'text_generation_mae','paired_text_generation_mae','reference_generation_mae','caption_response_l1')}
        | {'broad_mae': r['broad_text']['generation_mae']} for r in history])
    body = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2000张文生图回放评估</title><style>body{font:16px/1.65 system-ui;background:#111827;color:#e5e7eb;max-width:1180px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%}details{background:#1f2937;border-radius:10px;padding:14px;margin:16px 0}summary{cursor:pointer}</style><h1>2000张文生图回放：训练内外对比</h1><p>比较训练前、文字最佳（新增4000步）、参考最佳（新增2000步）、最终（新增8000步）。综合最佳实际为训练起点，不冒充新权重。原生输出16×16；仅近邻放大展示。</p><p>参考测试56个、额外验证32个、参考训练24个；另有48个新广域验证、48个回放训练目标。全部两个固定种子。广域验证排除训练期间使用的1024个验证目标。测试不参与选版。</p><p><a href="/five-task-broad-replay/">训练阶段验证</a> · <a href="/five-task-broad-replay-evaluation/">原自动三版本评估</a></p>'''
    for prefix, title in [('test_', '参考图测试集'), ('val_', '参考图额外验证'),
                          ('train_', '参考图训练集'), ('broad_val_', '新广域文生图验证'),
                          ('replay_train_', '2000张回放训练抽样')]:
        group = [name for name in panels if name.startswith(prefix)]
        body += '<details'+(' open' if prefix == 'test_' else '')+'><summary>'+title+'</summary>'
        for name in group:
            body += '<p>'+html.escape(name)+'</p><a href="panels/'+name+'"><img loading="lazy" src="panels/'+name+'"></a>'
        body += '</details>'
    body += '<p>文字引起变化不等于修改正确。像素误差不能代替MC风格、可辨认性及装饰保留的人工判断。</p>'
    body += '<h2>常见文生图回归</h2>'
    for seed in range(2):
        body += f'<img loading="lazy" src="panels/common_s{seed}.png">'
    (out/'index.html').write_text(body)
    publish.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out/'panels', publish/'panels', dirs_exist_ok=True)
    shutil.copy2(out/'index.html', publish/'index.html')
    expected_reference = {f'{r["case_id"]}_{v}_{n}_s{s}.png' for r in rows for v in VARIANTS for n,_,_ in TASK_VIEWS for s in range(2)}
    expected_reference |= {f'common_{i:02d}_{v}_s{s}.png' for i in range(len(COMMON)) for v in VARIANTS for s in range(2)}
    expected_broad = {f'{r["case_id"]}_{v}_s{s}.png' for r in read(broad/'cases.json') for v in VARIANTS for s in range(2)}
    for directory, names in [(out/'native', expected_reference), (broad/'native', expected_broad)]:
        if {p.name for p in directory.glob('*.png')} != names:
            raise ValueError('Unexpected or missing evaluation output')
        for name in names:
            with Image.open(directory/name) as im:
                if im.size != (16,16) or im.mode != 'RGBA':
                    raise ValueError('Output grid or alpha changed')
    atomic_json(out/'verification.json', {'complete':True, 'reference_outputs':len(expected_reference),
        'broad_outputs':len(expected_broad), 'panels':len(panels), 'all_source_checkpoints_unchanged':True,
        'native_size':16, 'all_rgba_valid':True, 'no_test_selection':True,
        'public_gallery':str(publish), 'url':'http://localhost:8766/replay-evaluation/'})
    print('EVALUATION COMPLETE', len(expected_reference)+len(expected_broad), 'images',len(panels),'panels',flush=True)


def run(root, base, out, device):
    prepare(root,base,out)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.4,device)
    shared.VARIANTS=VARIANTS
    for variant in ('text_best','reference_best'):
        if not (out/f'progress_{variant}.json').exists() or read(out/f'progress_{variant}.json').get('phase')!='complete':
            shared.generate(out,variant,device)
            gc.collect();torch.cuda.empty_cache()
    if not (out/'broad/protocol.json').exists():
        broad_cases(root,out)
    for variant in VARIANTS:
        if not (out/'broad'/f'progress_{variant}.json').exists():
            broad_generate(out,variant,device)
            gc.collect();torch.cuda.empty_cache()
    for source,destination in [('previous','before'),('last','last')]:
        while not import_base(base,out,source,destination):
            state=read(root/'queue_status.json')
            if state['phase'] in ('complete','failed','stopped'):
                shared.generate(out,destination,device)
                break
            print('WAIT FOR AUTOMATIC EVALUATION',source,flush=True)
            time.sleep(10)
    render(root,out,Path('outputs/native16_data_review_20261107/replay-evaluation'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/reference_text_replay2000_20261108_balanced'))
    p.add_argument('--base',type=Path,default=Path('outputs/reference_text_replay2000_evaluation_20261108_balanced'))
    p.add_argument('--out',type=Path,default=Path('outputs/reference_text_replay2000_review_20261108'))
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    torch.set_num_threads(4)
    run(a.root,a.base,a.out,a.device)
