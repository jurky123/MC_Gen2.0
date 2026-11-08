"""Evaluate all five original-caption tasks on train, fresh val and full test."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData, TASK_VIEWS
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from evaluate_lossless_native16 import model_hash
from prepare_reference_recovery import COMMON
from train_five_task_reference import draw_sheet, fixed_noise
from train_visual_reference import thumbnail
from validate_resolution import image_metrics

DEFAULT = Path('outputs/reference_five_task_evaluation_20261108')
VARIANTS = ('previous', 'best', 'last')
SEEDS = (2026110841, 2026110842)


def read(path):
    return json.loads(path.read_text())


def prepare(root, out):
    if not read(root/'status.json')['complete']:
        raise ValueError('Final evaluation requires completed training')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'native', 'panels'):
        (out/name).mkdir()
    data = FiveTaskData(root)
    run = read(root/'run.json')
    monitored = set(run['validation_indices']) | set(run['generation_panel_indices'])
    monitored |= {i for pair in run['text_control_pairs'] for i in pair}
    monitored_hashes = {data.rows[i]['target_hash'] for i in monitored if data.rows[i]['split'] == 'val'}
    selected = []
    seen = set()

    def add(i):
        row = data.rows[i]
        if row['target_hash'] in seen:
            return
        seen.add(row['target_hash'])
        selected.append({**row, 'pair_index': i,
            'case_id': f'{row["split"]}_{sum(x["split"] == row["split"] for x in selected):02d}'})

    for i, row in enumerate(data.rows):
        if row['split'] == 'test':
            add(i)
    rng = random.Random(2026110919)
    for quality in ('highest_quality', 'evaluation_only'):
        groups = defaultdict(list)
        unique = {}
        for i, row in enumerate(data.rows):
            if row['split'] == 'val' and row['quality_tier'] == quality and row['target_hash'] not in monitored_hashes:
                unique.setdefault(row['target_hash'], i)
        for i in unique.values():
            groups[data.categories[i]].append(i)
        for values in groups.values():
            rng.shuffle(values)
        chosen = []
        while len(chosen) < 16:
            for category in sorted(groups):
                if groups[category] and len(chosen) < 16:
                    chosen.append(groups[category].pop())
            if not any(groups.values()) and len(chosen) < 16:
                raise ValueError('Not enough fresh validation targets')
        for i in chosen:
            add(i)
    previous = read(Path('outputs/reference_recovery_evaluation_20261108/samples.json'))
    for row in previous:
        if row['split'] == 'train':
            assert data.rows[row['pair_index']]['id'] == row['id']
            add(row['pair_index'])
    assert Counter(r['split'] for r in selected) == {'test': 56, 'val': 32, 'train': 24}
    overlaps = {}
    for key in ('target_hash', 'reference_sha256', 'split_group'):
        trained = {r[key] for r in data.rows if r['split'] == 'train'}
        overlaps[key] = len(trained & {r[key] for r in selected if r['split'] != 'train'})
    assert not any(overlaps.values())
    paths = {'previous': Path(read(root/'initialization.json')['checkpoint']),
             'best': root/'training/best_balanced.pt', 'last': root/'training/latest.pt'}
    models = {}
    for variant, path in paths.items():
        state = load_checkpoint(path)
        assert not state.get('diagnostic_only') and not state.get('target_leakage')
        models[variant] = {'checkpoint': str(path.resolve()), 'step': state['step'],
            'file_sha256': file_hash(path), 'model_sha256': model_hash(state['model']),
            'text_identity': state['text_identity'], 'visual_identity': state['visual_identity']}
        del state
        gc.collect()
    ids = [r['pair_index'] for r in selected]
    h, mask = data.encoder.encode(conditioned_prompts(COMMON, data.cfg), 'cpu')
    torch.save({'text': data.text[ids], 'mask': data.mask[ids],
        'reference': torch.from_numpy(np.array(data.features[ids], copy=True)).view(torch.bfloat16),
        'common': h, 'common_mask': mask}, out/'features.pt')
    for row in selected:
        with Image.open(row['reference_file']) as im:
            thumbnail(im, 256).save(out/'inputs'/f'{row["case_id"]}_reference.png')
        with Image.open(row['target_file']) as im:
            assert im.size == (16, 16)
            im.convert('RGBA').save(out/'inputs'/f'{row["case_id"]}_target.png')
    atomic_json(out/'samples.json', selected)
    atomic_json(out/'protocol.json', {'source_run': str(root.resolve()), 'source_pairs_sha256': data.pairs_sha256,
        'features_sha256': file_hash(out/'features.pt'), 'models': models,
        'counts': dict(Counter(r['split'] for r in selected)), 'train_holdout_overlap': overlaps,
        'all_test_targets_included': True, 'fresh_validation_excludes_all_training_monitors': True,
        'seeds': list(SEEDS), 'steps': 32, 'precision': 'float32', 'cfg': 1.,
        'views': [n for n, _, _ in TASK_VIEWS], 'short_is_original_caption': True,
        'test_used_for_checkpoint_selection': False, 'no_output_filtering': True,
        'holdout_scope': 'reference fine-tuning, not a claim about base pretraining'})
    print('PREPARED', dict(Counter(r['split'] for r in selected)), flush=True)


@torch.inference_mode()
def generate(out, variant, device):
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.4, device)
    protocol = read(out/'protocol.json')
    meta = protocol['models'][variant]
    assert file_hash(meta['checkpoint']) == meta['file_sha256']
    rows = read(out/'samples.json')
    # Exact duplicate weights reuse completed outputs instead of rerunning.
    for previous in VARIANTS[:VARIANTS.index(variant)]:
        if protocol['models'][previous]['model_sha256'] == meta['model_sha256'] and (out/f'progress_{previous}.json').exists() and read(out/f'progress_{previous}.json')['phase'] == 'complete':
            for path in (out/'native').glob(f'*_{previous}_*.png'):
                shutil.copy2(path, path.with_name(path.name.replace('_'+previous+'_', '_'+variant+'_')))
            atomic_json(out/f'progress_{variant}.json', {'phase': 'complete', 'exact_weight_alias': previous})
            return
    f = torch.load(out/'features.pt', weights_only=True)
    f = {k: v.to(device) for k, v in f.items()}
    state = load_checkpoint(meta['checkpoint'])
    assert state['text_identity'] == meta['text_identity'] and state['visual_identity'] == meta['visual_identity']
    cfg = Experiment.from_dict(state['config'])
    assert cfg.model.size == 16
    model = PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    del state
    gc.collect()
    started = time.monotonic()
    done = 0
    for si, seed in enumerate(SEEDS):
        for name, task, v in TASK_VIEWS:
            for start in range(0, len(rows), 16):
                ids = [i for i in range(start, min(start+16, len(rows)))
                       if not (out/'native'/f'{rows[i]["case_id"]}_{variant}_{name}_s{si}.png').exists()]
                if not ids:
                    continue
                mask = f['mask'][ids, v].clone()
                if task == 'reference':
                    mask.zero_()
                images = sample(model, f['text'][ids, v], mask,
                    f['reference'][ids] if task != 'text' else None,
                    seeds=[seed]*len(ids), steps=32, text_scale=1., ref_scale=1.,
                    initial_noise=fixed_noise(len(ids), seed, device))
                assert bool(torch.isfinite(images).all())
                for value, i in zip(images, ids):
                    Image.fromarray(decode(value)).save(out/'native'/f'{rows[i]["case_id"]}_{variant}_{name}_s{si}.png')
                done += len(ids)
                atomic_json(out/f'progress_{variant}.json', {'phase': 'running', 'mode': name,
                    'seed_index': si, 'done': done, 'seconds': time.monotonic()-started})
        common = sample(model, f['common'], f['common_mask'], seeds=[seed]*len(COMMON), steps=32,
            text_scale=2., initial_noise=fixed_noise(len(COMMON), seed, device))
        for j, value in enumerate(common):
            Image.fromarray(decode(value)).save(out/'native'/f'common_{j:02d}_{variant}_s{si}.png')
    atomic_json(out/f'progress_{variant}.json', {'phase': 'complete', 'seconds': time.monotonic()-started})
    print('COMPLETE', variant, done, flush=True)


def render(out, publish):
    protocol, rows = read(out/'protocol.json'), read(out/'samples.json')
    assert file_hash(out/'features.pt') == protocol['features_sha256']
    for variant in VARIANTS:
        assert read(out/f'progress_{variant}.json')['phase'] == 'complete'
        assert file_hash(protocol['models'][variant]['checkpoint']) == protocol['models'][variant]['file_sha256']
    expected = {f'{r["case_id"]}_{v}_{n}_s{s}.png' for r in rows for v in VARIANTS for n, _, _ in TASK_VIEWS for s in range(2)}
    expected |= {f'common_{j:02d}_{v}_s{s}.png' for j in range(len(COMMON)) for v in VARIANTS for s in range(2)}
    assert {p.name for p in (out/'native').glob('*.png')} == expected
    manifest = []
    for name in sorted(expected):
        with Image.open(out/'native'/name) as im:
            assert im.mode == 'RGBA' and im.size == (16, 16)
            im.load()
        manifest.append({'file': name, 'sha256': file_hash(out/'native'/name)})
    metrics = []
    for row in rows:
        with Image.open(out/'inputs'/f'{row["case_id"]}_target.png') as im:
            truth = np.array(im.convert('RGBA')).repeat(2, 0).repeat(2, 1)
        for v in VARIANTS:
            for n, _, _ in TASK_VIEWS:
                for s in range(2):
                    metrics.append({'case_id': row['case_id'], 'split': row['split'], 'variant': v,
                        'mode': n, 'seed_index': s, **image_metrics(out/'native'/f'{row["case_id"]}_{v}_{n}_s{s}.png', truth)})
    summary = []
    for split in ('test', 'val', 'train'):
        for v in VARIANTS:
            for n, _, _ in TASK_VIEWS:
                group = [x for x in metrics if (x['split'], x['variant'], x['mode']) == (split, v, n)]
                summary.append({'split': split, 'variant': v, 'mode': n, 'outputs': len(group),
                    **{k: float(np.mean([x[k] for x in group])) for k in ('premultiplied_mae', 'alpha_iou')}})
    atomic_json(out/'metrics.json', {'summary': summary, 'per_output': metrics,
        'caveat': 'Pixel alignment proxies; coarse descriptions may admit multiple valid targets.'})
    atomic_json(out/'generation_manifest.json', manifest)
    panels = []

    def raw(r, v, n, s=0):
        return out/'native'/f'{r["case_id"]}_{v}_{n}_s{s}.png'

    def inputs(r):
        return [out/'inputs'/f'{r["case_id"]}_reference.png', out/'inputs'/f'{r["case_id"]}_target.png']

    for split in ('test', 'val', 'train'):
        group = [r for r in rows if r['split'] == split]
        for page, start in enumerate(range(0, len(group), 8), 1):
            part = group[start:start+8]
            for n, _, _ in TASK_VIEWS:
                for s in range(2):
                    name = f'{split}_{n}_s{s}_{page:02d}.png'
                    entries = [(inputs(r)+[raw(r, v, n, s) for v in VARIANTS], r['case_id']+' | '+r['captions'][0],
                                'Same reference, caption and noise; FP32 native16') for r in part]
                    draw_sheet(out/'panels'/name, f'{split.upper()} | {n} | seed{s+1}',
                        ['HD reference', 'MC target', 'Previous8000', 'Balanced best', 'New final'], entries)
                    panels.append(name)
            for v in VARIANTS:
                name = f'{split}_{v}_modes_{page:02d}.png'
                entries = [(inputs(r)+[raw(r, v, n) for n in ('text_short', 'text_detail', 'joint_short', 'joint_detail', 'reference')],
                    r['case_id']+' | '+r['captions'][0], 'Distinct detail' if len(r['captions']) > 1 else 'Detail alias: original short label') for r in part]
                draw_sheet(out/'panels'/name, f'{split.upper()} | {v} | five views | seed A',
                    ['HD reference', 'MC target', 'Short T2I', 'Detail T2I', 'Short + image', 'Detail + image', 'Image only'], entries)
                panels.append(name)
    for s in range(2):
        name = f'common_s{s}.png'
        entries = [([out/'native'/f'common_{j:02d}_{v}_s{s}.png' for v in VARIANTS], prompt, 'Pure text | CFG2 | matched noise') for j, prompt in enumerate(COMMON)]
        draw_sheet(out/'panels'/name, 'Common T2I regression', ['Previous8000', 'Balanced best', 'New final'], entries)
        panels.append(name)
    payload = json.dumps({'panels': panels}, ensure_ascii=False)
    body = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五任务训练 · 训练集外评估</title><style>body{font:16px/1.6 system-ui;background:#111827;color:#e5e7eb;max-width:1120px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%}article{background:#1f2937;padding:16px;margin:18px 0;border-radius:10px}select{font:inherit;padding:8px;margin:8px}.controls{position:sticky;top:0;background:#1f2937}</style><p><a href="/five-task/">本轮训练验证与文字响应</a> · <a href="/recovery-evaluation/">上一轮评估</a></p><h1>五任务训练：训练集外评估</h1><p>全部测试56个、额外验证32个、训练24个；每个目标两个种子，未筛选生成结果。默认测试集。对照上一轮8000步、本轮综合验证最佳、本轮最终权重。粗标为原始名称，精标为已有MC描述；缺少精标的旧数据如实回退粗标。</p><div class="controls"><label>数据<select id="split"><option value="test">测试56</option><option value="val">额外验证32</option><option value="train">训练24</option></select></label><label>任务<select id="view"><option value="joint_short">粗标图文</option><option value="joint_detail">精标图文</option><option value="text_short">粗标文生图</option><option value="text_detail">精标文生图</option><option value="reference">纯参考图</option><option value="last_modes">最终权重五任务</option><option value="best_modes">综合最佳五任务</option><option value="previous_modes">上一轮五任务</option></select></label><select id="seed"><option value="0">种子A</option><option value="1">种子B（任务对比）</option></select></div><div id="items"></div><h2>常见文生图回归</h2>'''
    for s in range(2):
        body += f'<img loading="lazy" src="panels/common_s{s}.png">'
    body += '''<p>未见范围为本轮参考图微调。像素对齐指标不能代替MC风格、文字语义与装饰的人工审查。</p><script>const data=PAYLOAD;const split=document.getElementById('split'),view=document.getElementById('view'),seed=document.getElementById('seed'),items=document.getElementById('items');function show(){const key=split.value+'_'+view.value+(view.value.endsWith('_modes')?'_':'_s'+seed.value+'_');items.replaceChildren();data.panels.filter(p=>p.startsWith(key)).forEach(p=>{const a=document.createElement('article'),link=document.createElement('a'),im=document.createElement('img');link.href='panels/'+p;link.textContent=p;im.src=link.href;im.loading='lazy';a.append(link,document.createElement('br'),im);items.append(a)})}[split,view,seed].forEach(x=>x.addEventListener('change',show));show();</script>'''.replace('PAYLOAD', payload)
    (out/'index.html').write_text(body)
    js = re.search(r'<script>(.*?)</script>', body, re.S).group(1)
    check = out/'gallery_check.js'
    check.write_text(js)
    subprocess.run(['node', '--check', str(check)], check=True, capture_output=True)
    check.unlink()
    publish.mkdir(parents=True, exist_ok=True)
    for name in ('inputs', 'native', 'panels'):
        shutil.copytree(out/name, publish/name, dirs_exist_ok=True)
    temporary = publish/'index.html.tmp'
    shutil.copy2(out/'index.html', temporary)
    temporary.replace(publish/'index.html')
    atomic_json(out/'verification.json', {'complete': True, 'native_pngs': len(expected), 'panels': len(panels),
        'all_test_included': True, 'all_source_checkpoint_hashes_unchanged': True,
        'all_native_dimensions_and_rgba_valid': True, 'default_split': 'test', 'javascript_syntax': 'passed'})
    print('VERIFIED', len(expected), 'native images;', len(panels), 'panels', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'generate', 'render'))
    p.add_argument('--root', type=Path, default=Path('outputs/reference_five_task_20261108'))
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--publish', type=Path, default=Path('outputs/native16_data_review_20261107/five-task-evaluation'))
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare(args.root, args.out)
    elif args.stage == 'generate':
        if not args.variant:
            p.error('--variant required')
        generate(args.out, args.variant, args.device)
    else:
        render(args.out, args.publish)
