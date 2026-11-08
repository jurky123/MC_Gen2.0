"""Full held-out test gallery for the long-budget native16 reference run."""
import argparse
from collections import Counter, defaultdict
import gc
import html
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.reference_recovery import RecoveryData
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from evaluate_lossless_native16 import model_hash
from prepare_reference_recovery import COMMON
from train_visual_reference import thumbnail
from validate_resolution import image_metrics

RUN = Path('outputs/reference_recovery_20261108')
DEFAULT = Path('outputs/reference_recovery_evaluation_20261108')
SEEDS = [2026110841, 2026110842]
VARIANTS = ('before', 'best', 'last')
MODES = ('text', 'reference', 'joint', 'joint_short', 'joint_cfg2')


def read(path):
    return json.loads(path.read_text())


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('native', 'inputs', 'panels'):
        (out/name).mkdir()
    if not read(RUN/'status.json')['complete']:
        raise ValueError('Wait until training completes to fix checkpoints')
    data = RecoveryData(RUN)
    run = read(RUN/'run.json')
    monitor_hashes = {data.rows[i]['target_hash'] for i in run['validation_indices']}
    rng = random.Random(2026110840)
    selected = []
    seen = set()

    def add(i):
        row = data.rows[i]
        if row['target_hash'] in seen:
            return
        seen.add(row['target_hash'])
        ident = f'{row["split"]}_{sum(r["split"] == row["split"] for r in selected):02d}'
        selected.append({**row, 'pair_index': i, 'case_id': ident,
                         'category_condition': data.categories[i],
                         'seen_in_training_monitor': i in run['training_monitor_indices']})

    # Every held-out test target, not a favorable subset.
    for i, row in enumerate(data.rows):
        if row['split'] == 'test':
            add(i)

    def choose(split, quality, count):
        groups = defaultdict(list)
        unique = {}
        for i, row in enumerate(data.rows):
            if (row['split'], row['quality_tier']) != (split, quality):
                continue
            if row['target_hash'] in seen or (split == 'val' and row['target_hash'] in monitor_hashes):
                continue
            unique.setdefault(row['target_hash'], i)
        for i in unique.values():
            groups[data.categories[i]].append(i)
        for pool in groups.values():
            rng.shuffle(pool)
        result = []
        while len(result) < count:
            for category in sorted(groups):
                if groups[category] and len(result) < count:
                    result.append(groups[category].pop())
            if not any(groups.values()) and len(result) < count:
                raise ValueError(f'Not enough targets: {split}/{quality}')
        for i in result:
            add(i)

    choose('val', 'highest_quality', 16)
    choose('val', 'evaluation_only', 16)
    for i in run['training_monitor_indices']:
        add(i)
    choose('train', 'reviewed', 8)
    counts = dict(Counter(r['split'] for r in selected))
    if counts != {'test': 56, 'val': 32, 'train': 24}:
        raise ValueError(counts)
    leakage = {}
    for key in ('target_hash', 'reference_sha256', 'split_group'):
        trained = {r[key] for r in data.rows if r['split'] == 'train'}
        overlap = {r[key] for r in selected if r['split'] != 'train'} & trained
        leakage[key] = len(overlap)
    if any(leakage.values()):
        raise ValueError('Held-out evaluation leakage: ' + str(leakage))
    if any(r['target_hash'] in monitor_hashes for r in selected if r['split'] == 'val'):
        raise ValueError('Additional validation overlaps training monitoring')
    metadata = {}
    initialization = read(RUN/'initialization.json')
    for variant, path in [('before', Path(initialization['checkpoint'])),
                          ('best', RUN/'training/best_reference.pt'),
                          ('last', RUN/'training/latest.pt')]:
        state = load_checkpoint(path)
        if state.get('diagnostic_only') or state.get('target_leakage'):
            raise ValueError('Diagnostic checkpoint is not an evaluation model')
        metadata[variant] = {'checkpoint': str(path.resolve()), 'step': state['step'],
            'file_sha256': file_hash(path), 'model_sha256': model_hash(state['model']),
            'text_identity': state['text_identity'], 'visual_identity': state['visual_identity']}
        del state
        gc.collect()
    indices = [r['pair_index'] for r in selected]
    h, m = data.encoder.encode(conditioned_prompts(COMMON, data.cfg), 'cpu')
    torch.save({'coarse': data.coarse_text[indices], 'short': data.text[indices, 0],
        'coarse_mask': data.coarse_mask[indices], 'short_mask': data.mask[indices, 0],
        'reference': torch.from_numpy(np.array(data.features[indices], copy=True)).view(torch.bfloat16),
        'target': data.targets[indices], 'common': h, 'common_mask': m,
        'text_identity': data.encoder.identity, 'visual_identity': data.visual_identity}, out/'features.pt')
    for row in selected:
        with Image.open(row['reference_file']) as im:
            thumbnail(im, 256).save(out/'inputs'/f'{row["case_id"]}_reference.png')
        with Image.open(row['target_file']) as im:
            assert im.size == (16, 16)
            im.convert('RGBA').save(out/'inputs'/f'{row["case_id"]}_target.png')
    atomic_json(out/'samples.json', selected)
    atomic_json(out/'protocol.json', {'source_run': str(RUN.resolve()),
        'source_pairs_sha256': data.pairs_sha256, 'features_sha256': file_hash(out/'features.pt'),
        'models': metadata, 'counts': counts, 'unique_targets': len(seen),
        'all_test_targets_included': True, 'additional_validation_excludes_monitoring': True,
        'train_holdout_overlap': leakage, 'selection_seed': 2026110840,
        'generation_seeds': SEEDS, 'steps': 32, 'solver': 'heun', 'precision': 'float32',
        'primary_cfg': 1., 'secondary_joint_cfg': 2., 'modes': list(MODES),
        'text_condition': 'text and joint use exactly the same coarse category embedding; joint_short uses original short name',
        'noise': 'same CPU iid Gaussian16 for all targets, modes and checkpoints within a seed',
        'test_usage': 'evaluation only; checkpoints were fixed using validation before these test generations',
        'holdout_scope': 'unseen in reference fine-tuning; not a claim of exclusion from base pretraining',
        'common_prompts': COMMON, 'no_output_filtering': True, 'training_started': False})
    print('PREPARED', counts, 'overlap', leakage, flush=True)


def noise(n, seed, device):
    return torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(seed)).repeat(n, 1, 1, 1).to(device)


@torch.inference_mode()
def generate(out, variant, device):
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.10, device)
    protocol = read(out/'protocol.json')
    metadata = protocol['models'][variant]
    if file_hash(metadata['checkpoint']) != metadata['file_sha256']:
        raise ValueError('Evaluation checkpoint changed')
    f = torch.load(out/'features.pt', weights_only=True)
    if f['text_identity'] != metadata['text_identity'] or f['visual_identity'] != metadata['visual_identity']:
        raise ValueError('Condition encoder mismatch')
    state = load_checkpoint(metadata['checkpoint'])
    cfg = Experiment.from_dict(state['config'])
    assert cfg.model.size == 16
    model = PixelFlow(cfg.model).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    del state
    gc.collect()
    f = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in f.items()}
    rows = read(out/'samples.json')
    done = 0
    started = time.monotonic()
    for si, seed in enumerate(SEEDS):
        for mode in MODES:
            for start in range(0, len(rows), 16):
                ids = list(range(start, min(start+16, len(rows))))
                missing = [i for i in ids if not (out/'native'/f'{rows[i]["case_id"]}_{variant}_{mode}_s{si}.png').exists()]
                if not missing:
                    continue
                key = 'short' if mode == 'joint_short' else 'coarse'
                h, mask = f[key][missing], f[key+'_mask'][missing].clone()
                if mode == 'reference':
                    mask.zero_()
                refs = None if mode == 'text' else f['reference'][missing]
                scale = 2. if mode == 'joint_cfg2' else 1.
                out_images = sample(model, h, mask, refs, seeds=[seed]*len(missing), steps=32,
                    text_scale=scale, ref_scale=scale, initial_noise=noise(len(missing), seed, device))
                if not bool(torch.isfinite(out_images).all()):
                    raise FloatingPointError('Nonfinite generation')
                for value, i in zip(out_images, missing):
                    Image.fromarray(decode(value)).save(out/'native'/f'{rows[i]["case_id"]}_{variant}_{mode}_s{si}.png')
                done += len(missing)
                atomic_json(out/f'progress_{variant}.json', {'phase': 'running', 'seed': si,
                    'mode': mode, 'outputs_written': done, 'seconds': time.monotonic()-started})
        common = sample(model, f['common'], f['common_mask'], seeds=[seed]*len(COMMON),
            steps=32, text_scale=2., initial_noise=noise(len(COMMON), seed, device))
        for j, value in enumerate(common):
            Image.fromarray(decode(value)).save(out/'native'/f'common_{j:02d}_{variant}_s{si}.png')
    # At fixed noisy input, compare matched reference against another design
    # from the same split and category. Singletons excluded from wrong-ref means.
    groups = defaultdict(list)
    for j, row in enumerate(rows):
        groups[(row['split'], row['category_condition'])].append(j)
    wrong = list(range(len(rows)))
    paired = []
    for pool in groups.values():
        if len(pool) > 1:
            paired += pool
            for i, j in zip(pool, pool[1:]+pool[:1]):
                wrong[i] = j
    flow = []
    for mode in ('joint', 'reference'):
        mask = f['coarse_mask'] if mode == 'joint' else torch.zeros_like(f['coarse_mask'])
        for level in (.1, .5, .9, .99):
            eps = noise(len(rows), SEEDS[0], device)
            x = (1-level)*f['target']+level*eps
            record = {'mode': mode, 'level': level, 'same_category_indices': sorted(paired), 'scores': {}}
            for label, refs in [('matched', f['reference']), ('wrong', f['reference'][wrong]), ('none', None)]:
                values = []
                for start in range(0, len(rows), 16):
                    stop = min(start+16, len(rows))
                    ctx = model.context(f['coarse'][start:stop], mask[start:stop], None if refs is None else refs[start:stop], None)
                    prediction = model(x[start:stop], torch.full((stop-start,), level, device=device), ctx)
                    values += (prediction-(eps-f['target'])[start:stop]).square().mean((1, 2, 3)).cpu().tolist()
                record['scores'][label] = values
            flow.append(record)
    atomic_json(out/f'flow_{variant}.json', flow)
    atomic_json(out/f'progress_{variant}.json', {'phase': 'complete', 'outputs_written': done,
        'seconds': time.monotonic()-started})
    print('COMPLETE', variant, done, round(time.monotonic()-started, 1), flush=True)


def render(out, publish):
    protocol = read(out/'protocol.json')
    rows = read(out/'samples.json')
    assert file_hash(out/'features.pt') == protocol['features_sha256']
    assert file_hash(RUN/'pairs.json') == protocol['source_pairs_sha256']
    for variant in VARIANTS:
        assert read(out/f'progress_{variant}.json')['phase'] == 'complete'
        meta = protocol['models'][variant]
        assert file_hash(meta['checkpoint']) == meta['file_sha256']
    expected = {f'{r["case_id"]}_{v}_{m}_s{s}.png' for r in rows for v in VARIANTS for m in MODES for s in range(2)}
    expected |= {f'common_{j:02d}_{v}_s{s}.png' for j in range(len(COMMON)) for v in VARIANTS for s in range(2)}
    assert {p.name for p in (out/'native').glob('*.png')} == expected
    manifest = []
    for name in sorted(expected):
        path = out/'native'/name
        with Image.open(path) as im:
            assert im.mode == 'RGBA' and im.size == (16, 16)
            im.load()
        manifest.append({'file': name, 'sha256': file_hash(path), 'native_size': 16})
    per_output = []
    for row in rows:
        with Image.open(out/'inputs'/f'{row["case_id"]}_target.png') as im:
            target = np.array(im.convert('RGBA')).repeat(2, 0).repeat(2, 1)
        for v in VARIANTS:
            for m in MODES:
                for s in range(2):
                    path = out/'native'/f'{row["case_id"]}_{v}_{m}_s{s}.png'
                    per_output.append({'case_id': row['case_id'], 'split': row['split'],
                        'quality': row['quality_tier'], 'variant': v, 'mode': m, 'seed_index': s,
                        **image_metrics(path, target)})
    summaries = []
    for split in ('test', 'val', 'train'):
        for quality in (None, 'highest_quality', 'reviewed', 'evaluation_only'):
            for v in VARIANTS:
                for m in MODES:
                    group = [x for x in per_output if (x['split'], x['variant'], x['mode']) == (split, v, m)
                             and (quality is None or x['quality'] == quality)]
                    if group:
                        summaries.append({'split': split, 'quality': quality or 'all', 'variant': v, 'mode': m,
                            'outputs': len(group), **{k: float(np.mean([x[k] for x in group])) for k in
                            ('premultiplied_mae', 'alpha_iou', 'largest_component_fraction', 'near_empty')}})
    flow_summary = []
    for v in VARIANTS:
        for record in read(out/f'flow_{v}.json'):
            for split in ('test', 'val', 'train'):
                ids = [j for j, r in enumerate(rows) if r['split'] == split]
                same = [j for j in ids if j in record['same_category_indices']
                        and rows[j]['category_condition'] != 'item']
                scores = record['scores']
                entry = {'variant': v, 'mode': record['mode'], 'level': record['level'], 'split': split,
                    'matched': float(np.mean([scores['matched'][j] for j in ids])),
                    'no_reference': float(np.mean([scores['none'][j] for j in ids])), 'same_category_count': len(same)}
                if same:
                    entry['same_category_wrong_minus_matched'] = float(np.mean([scores['wrong'][j]-scores['matched'][j] for j in same]))
                flow_summary.append(entry)
    regression = {}
    for v in ('best', 'last'):
        drift = []
        for j in range(len(COMMON)):
            for s in range(2):
                with Image.open(out/'native'/f'common_{j:02d}_before_s{s}.png') as im:
                    a = np.array(im).astype(np.int16)
                with Image.open(out/'native'/f'common_{j:02d}_{v}_s{s}.png') as im:
                    b = np.array(im).astype(np.int16)
                drift.append(np.abs(a-b))
        regression[v] = {'exact_pixels_equal': all(not x.any() for x in drift),
            'mean_uint8_drift': float(np.mean(drift)), 'max_uint8_drift': int(np.max(drift))}
    atomic_json(out/'metrics.json', {'summary': summaries, 'per_output': per_output,
        'flow_controls': flow_summary, 'common_text_regression': regression,
        'caveat': 'Contour/target alignment proxies, not decoration or MC style scores. Generic item prompts excluded from same-category flow controls.'})
    atomic_json(out/'generation_manifest.json', manifest)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    panels = []

    def raw(row, v, m, seed=0):
        return out/'native'/f'{row["case_id"]}_{v}_{m}_s{seed}.png'

    def make(name, title, labels, group, sources):
        canvas = Image.new('RGB', (14+154*len(labels), 66+164*len(group)), '#111827')
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), title, font=font, fill='white')
        for c, label in enumerate(labels):
            draw.text((14+c*154, 34), label, font=font, fill='#cbd5e1')
        for j, row in enumerate(group):
            y = 62+j*164
            paths = sources(row)
            for c, path in enumerate(paths):
                with Image.open(path) as im:
                    canvas.paste(thumbnail(im, 128, native=im.size == (16, 16)), (14+c*154, y))
            draw.text((14, y+132), f'{row["case_id"]} | {row["captions"][0]}'[:115], font=font, fill='white')
            draw.text((14, y+147), f'Class: {row["category_condition"]} | {row["quality_tier"]} | {row["split"]}', font=font, fill='#aebbd0')
        canvas.save(out/'panels'/name)
        panels.append(name)

    def inputs(r):
        return [out/'inputs'/f'{r["case_id"]}_reference.png', out/'inputs'/f'{r["case_id"]}_target.png']

    for split in ('test', 'val', 'train'):
        group = [r for r in rows if r['split'] == split]
        for page, start in enumerate(range(0, len(group), 8), 1):
            part = group[start:start+8]
            for s in range(2):
                make(f'{split}_compare_s{s}_{page:02d}.png', f'{split.upper()} | same class text + reference | seed {s+1}',
                    ['HD reference', 'MC16 target', 'Previous270', 'Best1000', 'Final8000'], part,
                    lambda r, seed=s: inputs(r)+[raw(r, v, 'joint', seed) for v in VARIANTS])
            for v in ('best', 'last'):
                make(f'{split}_{v}_modes_{page:02d}.png', f'{split.upper()} | {v.upper()} | class text and reference control',
                    ['HD reference', 'MC16 target', 'Class text only', 'Reference only', 'Class + ref A', 'Class + ref B'], part,
                    lambda r, v=v: inputs(r)+[raw(r, v, m) for m in ('text', 'reference', 'joint')]+[raw(r, v, 'joint', 1)])
            make(f'{split}_conditions_{page:02d}.png', f'{split.upper()} | Best1000 | text and CFG alternatives',
                ['HD reference', 'MC16 target', 'Class + ref CFG1', 'Name + ref CFG1', 'Class + ref CFG2'], part,
                lambda r: inputs(r)+[raw(r, 'best', m) for m in ('joint', 'joint_short', 'joint_cfg2')])
            make(f'{split}_last_conditions_{page:02d}.png', f'{split.upper()} | Final8000 | text and CFG alternatives',
                ['HD reference', 'MC16 target', 'Class + ref CFG1', 'Name + ref CFG1', 'Class + ref CFG2'], part,
                lambda r: inputs(r)+[raw(r, 'last', m) for m in ('joint', 'joint_short', 'joint_cfg2')])
    for category in ('helmet', 'sword', 'axe', 'shield'):
        group = [r for r in rows if r['split'] != 'train' and r['category_condition'] == category][:4]
        if len(group) < 2:
            continue
        make(f'heldout_fixed_{category}.png', f'HELD-OUT ONLY | identical prompt: {category} | identical noise',
            ['HD reference', 'MC16 target', 'Previous270', 'Best1000', 'Final8000', 'Best text only'], group,
            lambda r: inputs(r)+[raw(r, v, 'joint') for v in VARIANTS]+[raw(r, 'best', 'text')])
    for s in range(2):
        group = [{'case_id': str(j), 'captions': [p], 'category_condition': p, 'quality_tier': 'common', 'split': 'text'} for j, p in enumerate(COMMON)]
        make(f'common_s{s}.png', 'Text-only regression | common prompts | CFG2', ['Previous270', 'Best1000', 'Final8000'], group,
            lambda r, seed=s: [out/'native'/f'common_{int(r["case_id"]):02d}_{v}_s{seed}.png' for v in VARIANTS])
    build_html(out, rows, panels, summaries)
    script = re.search(r'<script>(.*?)</script>', (out/'index.html').read_text(), re.S).group(1)
    check = out/'gallery_check.js'
    check.write_text(script)
    subprocess.run(['node', '--check', str(check)], check=True, capture_output=True, text=True)
    check.unlink()
    atomic_json(out/'verification.json', {'complete': True, 'counts': protocol['counts'],
        'native_pngs_verified': len(manifest), 'unique_targets': len(rows), 'panels': len(panels),
        'all_native_dimensions_and_rgba_valid': True, 'train_holdout_overlap': protocol['train_holdout_overlap'],
        'all_test_targets_included': True, 'checkpoint_hashes_unchanged': True,
        'gallery_javascript_syntax': 'passed', 'gallery_default_split': 'test'})
    if publish:
        publish.mkdir(parents=True, exist_ok=True)
        for directory in ('native', 'inputs', 'panels'):
            shutil.copytree(out/directory, publish/directory, dirs_exist_ok=True)
        temporary = publish/'index.html.tmp'
        shutil.copy2(out/'index.html', temporary)
        temporary.replace(publish/'index.html')
    print('VERIFIED', len(manifest), 'native PNGs;', len(panels), 'panels', flush=True)


def build_html(out, rows, panels, summaries):
    data = json.dumps({'rows': [{'id': r['case_id'], 'split': r['split'], 'name': r['captions'][0],
                                'category': r['category_condition']} for r in rows], 'panels': panels}, ensure_ascii=False).replace('<', '\\u003c')
    content = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练集外评估 · 参考图生成</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:1100px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%;image-rendering:pixelated}button,select{font:inherit;padding:8px;margin:4px;background:#263447;color:white;border:1px solid #53647c;border-radius:6px}.controls{position:sticky;top:0;background:#1f2937;padding:12px;border-radius:10px}article,details{padding:16px;background:#1f2937;margin:18px 0;border-radius:10px}table{border-collapse:collapse}td,th{padding:6px 16px;border-bottom:1px solid #53647c;text-align:left}h2{font-size:20px}summary{cursor:pointer}</style>
<p><a href="/recovery/">训练阶段对照</a> · <a href="/evaluation/">上一轮测评</a> · <a href="/">数据查看</a></p>
<h1>参考图生成：训练集外评估</h1><p><b>测试集56个（全量） · 额外验证集32个 · 训练集24个；每个目标两个固定种子。</b>页面默认展示测试集。测试目标不参与本轮训练或权重选择；额外验证目标也未出现在本轮训练监控面板中。已核对与训练集的目标、参考图及划分组均无重叠。</p>
<p>比较上一轮270步、本轮验证最佳1000步、最终8000步。主要使用相同类别词与噪声，避免用精标文字代替参考图。所有输出均为原生16×16，展示图仅用最近邻放大，未挑选生成结果。</p>
<div class="controls"><label>数据<select id="split"><option value="test">测试集：全部56个</option><option value="val">额外验证集：32个</option><option value="train">训练集：24个</option></select></label><label>对照<select id="view"><option value="compare">三组权重对比</option><option value="best_modes">验证最佳：三种条件</option><option value="last_modes">最终权重：三种条件</option><option value="conditions">验证最佳：类别词/短标签/CFG</option><option value="last_conditions">最终权重：类别词/短标签/CFG</option></select></label><label>种子<select id="seed"><option value="0">种子A</option><option value="1">种子B（权重对比）</option></select></label></div><div id="items"></div>
<h2>训练集外：同一文字，更换参考设计</h2><p>以下均为测试或额外验证样本，相同类别词、相同噪声。末列纯文字输出用于判断图像究竟补充了哪些设计。</p>'''
    for p in panels:
        if p.startswith('heldout_fixed_'):
            content += f'<article><a href="panels/{p}">{html.escape(p)}</a><br><img loading="lazy" src="panels/{p}"></article>'
    content += '<details><summary>数值摘要：类别词加参考图（CFG1）</summary><p>MAE越低、轮廓IoU越高越好。这些仅衡量与配对目标的对齐，不能代替装饰与MC风格的人工审查。</p><table><tr><th>数据</th><th>权重</th><th>RGBA误差</th><th>轮廓IoU</th></tr>'
    labels = {'before': '上一轮270', 'best': '最佳1000', 'last': '最终8000'}
    for x in summaries:
        if x['quality'] == 'all' and x['mode'] == 'joint':
            content += f'<tr><td>{x["split"]}</td><td>{labels[x["variant"]]}</td><td>{x["premultiplied_mae"]:.4f}</td><td>{x["alpha_iou"]:.4f}</td></tr>'
    content += '</table></details><details><summary>常见文生图回归</summary>'
    for s in range(2):
        content += f'<p>种子{s+1}</p><img loading="lazy" src="panels/common_s{s}.png">'
    content += '</details><p>“训练集外”指未参与本轮参考图训练；不据此推断基础文生图预训练从未见过相关MC原图。</p>'
    content += '''<script>const data=PAYLOAD;const split=document.getElementById('split'),view=document.getElementById('view'),seed=document.getElementById('seed'),items=document.getElementById('items');function show(){const key=split.value+'_'+(view.value==='compare'?'compare_s'+seed.value:view.value)+'_';items.replaceChildren();data.panels.filter(p=>p.startsWith(key)).forEach(p=>{const a=document.createElement('article'),link=document.createElement('a'),im=document.createElement('img');link.href='panels/'+p;link.textContent=p;im.src=link.href;im.loading='lazy';a.append(link,document.createElement('br'),im);items.append(a)});const d=document.createElement('details'),s=document.createElement('summary');s.textContent='逐张原生输出（16×16）';d.append(s);data.rows.filter(r=>r.split===split.value).forEach(r=>{const p=document.createElement('p');p.textContent=r.id+' · '+r.name+' · 实际类别词：'+r.category+' ';['before','best','last'].forEach(v=>{const a=document.createElement('a');a.href='native/'+r.id+'_'+v+'_joint_s'+seed.value+'.png';a.textContent=v+' ';p.append(a)});d.append(p)});items.append(d)}[split,view,seed].forEach(x=>x.addEventListener('change',show));show();</script>'''.replace('PAYLOAD', data)
    (out/'index.html').write_text(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'generate', 'render'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--publish', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
    elif args.stage == 'generate':
        if not args.variant:
            parser.error('--variant is required')
        generate(args.out, args.variant, args.device)
    else:
        render(args.out, args.publish)
