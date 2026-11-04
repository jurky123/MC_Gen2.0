"""Isolated evaluation only: never modifies training, source, or checkpoints."""
import argparse
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import content_hash, encode, decode
from mcgen.model import PixelFlow
from mcgen.reference_adapter import backbone_digest
from mcgen.runtime import load_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from mcgen.visual_tokens import VisualTokenEncoder

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/reference_adapter_20261104'
NEW = ROOT/'outputs/spatial_reference_20261104'
SOURCE = ROOT/'outputs/reference_adapter_expanded_20261104'
OUT = ROOT/'outputs/spatial_vs_mlp_eval_20261104'
DEVICE = 'cuda:1'
SEEDS = [202611042, 202611043]
START = time.monotonic()


def progress(phase, **extra):
    value = {'phase': phase, 'seconds': time.monotonic()-START, **extra}
    atomic_json(OUT/'progress.json', value)
    print(json.dumps(value), flush=True)


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def preview(im, size=112, native=False):
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, '#edf0f4')
    bg.alpha_composite(im)
    tile = ImageOps.contain(bg.convert('RGB'), (size, size),
                           Image.Resampling.NEAREST if native else Image.Resampling.LANCZOS)
    result = Image.new('RGB', (size, size), '#edf0f4')
    result.paste(tile, ((size-tile.width)//2, (size-tile.height)//2))
    return result


def panel(rows, name, labels, suffixes, seed=0):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    canvas = Image.new('RGB', (20+126*len(labels), 60+142*len(rows)), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f'Old pooled MLP vs new spatial attention | {name} | seed {SEEDS[seed]}', fill='white', font=font)
    for c, label in enumerate(labels):
        draw.text((12+c*126, 31), label, fill='#cbd5e1', font=font)
    for j, row in enumerate(rows):
        y = 57+142*j
        paths = [OUT/'inputs'/f'{row["case_id"]}_{key}.png' if key in ('reference', 'target')
                 else OUT/'native'/f'{row["case_id"]}_{key}_s{seed}.png' for key in suffixes]
        for c, path in enumerate(paths):
            with Image.open(path) as im:
                canvas.paste(preview(im, native=suffixes[c] != 'reference'), (12+c*126, y))
        draw.text((12, y+115), f'{row["case_id"]} | {row["prompt"]} | {row["reference_generator"]}', fill='white', font=font)
    canvas.save(OUT/name)


def metrics(a, b):
    a = a.astype(np.float32)/255
    b = b.astype(np.float32)/255
    aa, ba = a[..., 3:], b[..., 3:]
    pa = np.concatenate((a[..., :3]*aa, aa), -1)
    pb = np.concatenate((b[..., :3]*ba, ba), -1)
    ma, mb = aa[..., 0] >= .5, ba[..., 0] >= .5
    union = np.logical_or(ma, mb).sum()
    return {'premultiplied_rgba_mae': float(np.abs(pa-pb).mean()),
            'alpha_iou': float(np.logical_and(ma, mb).sum()/union) if union else 1.,
            'foreground_pixels': int(ma.sum()), 'near_empty': bool(ma.sum() < 4)}


def prepare():
    if (OUT/'samples.json').exists():
        rows = json.loads((OUT/'samples.json').read_text())
        primary = [r for r in rows if r['split'] in ('train', 'test')]
        auxiliary = [r for r in rows if r.get('same_text_group')]
        pairs = json.loads((NEW/'pairs.json').read_text())
        lookup = {(r['target_hash'], r['reference_sha256']): i for i, r in enumerate(pairs)}
        return primary, auxiliary, pairs, lookup
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'inputs').mkdir()
    (OUT/'native').mkdir()
    primary = json.loads((SOURCE/'samples.json').read_text())
    pairs = json.loads((NEW/'pairs.json').read_text())
    assert file_hash(NEW/'pairs.json') == file_hash(OLD/'pairs.json')
    pair_lookup = {(r['target_hash'], r['reference_sha256']): i for i, r in enumerate(pairs)}
    train = {r['target_hash'] for r in pairs if r['split'] == 'train'}
    val = {r['target_hash'] for r in pairs if r['split'] == 'val'}
    assert not any(r['target_hash'] in train|val for r in primary if r['split'] == 'test')
    groups = defaultdict(dict)
    for i, r in enumerate(pairs):
        if r['split'] == 'val':
            groups[r['prompt']].setdefault(r['target_hash'], i)
    auxiliary = []
    for group_number, (prompt, variants) in enumerate(sorted(groups.items())):
        if len(variants) < 2:
            continue
        for variant, pair_index in enumerate(list(variants.values())[:2]):
            row = pairs[pair_index]
            auxiliary.append({**row, 'case_id': f'same_{group_number:03d}_{variant}',
                              'same_text_group': prompt, 'pair_index': pair_index})
    rows = primary+auxiliary
    maps = {}
    for row in rows:
        if file_hash(row['reference_file']) != row['reference_sha256']:
            raise ValueError('Reference changed')
        with Image.open(row['reference_file']) as im:
            ImageOps.contain(im.convert('RGBA'), (512, 512), Image.Resampling.LANCZOS).save(
                OUT/'inputs'/f'{row["case_id"]}_reference.png')
        if row in primary:
            target = np.array(Image.open(SOURCE/'inputs'/f'{row["case_id"]}_target.png').convert('RGBA'))
        else:
            spec = row['target']
            if spec['path'] not in maps:
                maps[spec['path']] = np.memmap(spec['path'], mode='r', dtype='uint8', shape=tuple(spec['shape']))
            target = np.array(maps[spec['path']][spec['index']])
        assert content_hash(target) == row['target_hash']
        Image.fromarray(target).save(OUT/'inputs'/f'{row["case_id"]}_target.png')
    atomic_json(OUT/'samples.json', rows)
    atomic_json(OUT/'protocol.json', {'primary_targets': len(primary), 'primary_counts': dict(Counter(r['split'] for r in primary)),
        'same_text_validation_groups': len(auxiliary)//2, 'same_text_targets': len(auxiliary),
        'primary_source': str(SOURCE), 'source_sha256': file_hash(SOURCE/'samples.json'),
        'pairs_sha256': file_hash(NEW/'pairs.json'), 'generation_seeds': SEEDS, 'steps': 32, 'solver': 'heun',
        'old_guidance': 'pooled image or joint embedding CFG=2', 'new_guidance': 'independent text=2, reference=2',
        'selection': 'All 64 train and 64 prior test cases, plus every validation caption with >=2 distinct targets; no output filtering.',
        'test_overlap_with_reference_train_or_val': False,
        'test_status': 'Previously reviewed fixed test set, not an untouched new benchmark; not used for checkpoint selection.',
        'budget_caveat': 'Old and new released/experimental checkpoints have different training exposure and task mixtures; this is a version comparison, not a controlled architecture ablation.',
        'old_outputs': 'Reuse the verified previous 32-step/seed outputs for primary samples; reproduce one original batch before accepting them.',
        'preview_inputs': 'Up to512px copies for display only; all encoders use the original verified HD files.'})
    return primary, auxiliary, pairs, pair_lookup


def features(primary, auxiliary, pairs, pair_lookup):
    if (OUT/'features.pt').exists():
        return torch.load(OUT/'features.pt', weights_only=True)
    old_meta = json.loads((SOURCE/'results.json').read_text())
    cached = torch.load(SOURCE/'embeddings.pt', weights_only=True)
    old_cache_meta = json.loads((OLD/'features/metadata.json').read_text())
    spatial_meta = json.loads((NEW/'visual_features/metadata.json').read_text())
    assert spatial_meta['complete']
    spatial = np.memmap(NEW/'visual_features/features.uint16.mmap', mode='r', dtype='uint16', shape=tuple(spatial_meta['shape']))
    cfg = Experiment.load(NEW/'config.yaml')
    text_encoder = DiskTextEncoder(ROOT/'outputs/qwen_vl_embedding_large_20261103/features')
    text, _ = text_encoder.encode(conditioned_prompts([r['prompt'] for r in auxiliary], cfg), 'cpu')
    result = {k: cached[k][:len(primary)].clone() for k in ('text', 'image', 'joint')}
    chunks = {}
    for mode in ('image', 'joint'):
        extra = []
        for row in auxiliary:
            i = row['pair_index']
            start = i//64*64
            name = f'{start:06d}.pt'
            if start not in chunks:
                path = OLD/'features'/name
                assert file_hash(path) == old_cache_meta['chunks'][name]
                chunks[start] = torch.load(path, weights_only=True)
            extra.append(chunks[start][mode][i-start])
        result[mode] = torch.cat((result[mode], torch.stack(extra)))
    result['text'] = torch.cat((result['text'], text))
    all_rows = primary+auxiliary
    new = torch.empty(len(all_rows), 64, 4096, dtype=torch.bfloat16)
    missing = []
    verified_chunks = set()
    for j, row in enumerate(all_rows):
        i = pair_lookup.get((row['target_hash'], row['reference_sha256']))
        if i is None:
            missing.append(j)
            continue
        start = i//64*64
        if start not in verified_chunks:
            assert hashlib.sha256(np.asarray(spatial[start:start+64]).tobytes()).hexdigest() == spatial_meta['chunks'][str(start)]
            verified_chunks.add(start)
        new[j] = torch.from_numpy(np.array(spatial[i])).view(torch.bfloat16)
    encoder = VisualTokenEncoder(cfg.text, DEVICE)
    assert encoder.identity == spatial_meta['visual_identity']
    for offset in range(0, len(missing), 16):
        ids = missing[offset:offset+16]
        images = []
        for i in ids:
            with Image.open(all_rows[i]['reference_file']) as im:
                images.append(im.convert('RGBA'))
        new[ids] = encoder.encode(images)
        progress('encoding_unseen_spatial_tokens', done=offset+len(ids), total=len(missing))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    result['spatial'] = new
    result['visual_identity'] = spatial_meta['visual_identity']
    result['multimodal_identity'] = old_meta['multimodal_identity']
    torch.save(result, OUT/'features.pt')
    return result


@torch.inference_mode()
def run_generation(primary, auxiliary, f):
    while not json.loads((NEW/'status.json').read_text()).get('complete', False):
        queue = json.loads((NEW/'queue_status.json').read_text())
        if queue['phase'] == 'failed':
            raise RuntimeError('Training failed; do not interfere with the parent task')
        progress('waiting_for_training_completion', training=json.loads((NEW/'status.json').read_text())['step'])
        time.sleep(5)
    snapshots = {}
    models = {}
    for name, path in [('old', OLD/'training/best_flow.pt'), ('new', NEW/'training/best_joint_flow.pt'),
                       ('new_last', NEW/'training/latest.pt')]:
        state = load_checkpoint(path)
        cfg = Experiment.from_dict(state['config'])
        model = PixelFlow(cfg.model).to(DEVICE).eval()
        model.load_state_dict(state['model'], strict=True)
        models[name] = model
        snapshots[name] = {'path': str(path), 'step': state['step'], 'sha256': file_hash(path),
                           'exposures': state['step']*cfg.train.batch_size/5716,
                           'batch_size': cfg.train.batch_size,
                           'joint_fraction': cfg.train.task_mix['joint']}
        if name == 'old':
            assert state['multimodal_identity'] == f['multimodal_identity']
        else:
            assert state['visual_identity'] == f['visual_identity']
        del state
    assert len({backbone_digest(m) for m in models.values()}) == 1
    base_state = load_checkpoint(ROOT/'checkpoints/mcgen-step4128.pt')
    original = PixelFlow(Experiment.from_dict(base_state['config']).model).to(DEVICE).eval()
    original.load_state_dict(base_state['model'], strict=True)
    del base_state
    atomic_json(OUT/'checkpoints.json', snapshots)
    rows = primary+auxiliary
    written = 0

    def infer(name, mode, ids, seed):
        model = models[name]
        text = f['text'][ids].to(DEVICE)
        mask = torch.ones(text.shape[:2], dtype=torch.bool, device=DEVICE)
        if mode == 'text':
            return sample(model, text, mask, seeds=[seed]*len(ids), steps=32, text_scale=2.)
        if name == 'old':
            h = f[mode][ids].to(DEVICE)
            return sample(model, h, mask, seeds=[seed]*len(ids), steps=32, text_scale=2., embedding_adapter=True)
        if mode == 'image':
            mask = torch.zeros_like(mask)
        return sample(model, text, mask, f['spatial'][ids].to(DEVICE), seeds=[seed]*len(ids), steps=32,
                      text_scale=2., ref_scale=2.)

    # Reproduce the previous original batch, preserving its batch shape and ordering.
    ids = list(range(32))
    regenerated = infer('old', 'joint', ids, SEEDS[0])
    errors = []
    for i, value in zip(ids, regenerated):
        previous = np.array(Image.open(SOURCE/'native'/f'{primary[i]["case_id"]}_joint_s0.png'))
        errors.append(int(np.abs(decode(value).astype(int)-previous.astype(int)).max()))
    if max(errors) > 1:
        raise RuntimeError('Published old outputs did not reproduce within one quantization level')
    plain = infer('new', 'text', ids, SEEDS[0])
    h = f['text'][ids].to(DEVICE)
    baseline = sample(original, h, torch.ones(h.shape[:2], dtype=torch.bool, device=DEVICE),
                      seeds=[SEEDS[0]]*len(ids), steps=32, text_scale=2.)
    assert torch.equal(plain, baseline), 'New text differs from the original under identical execution conditions'
    text_errors = [int(np.abs(decode(value).astype(int)-np.array(Image.open(SOURCE/'native'/f'{primary[i]["case_id"]}_text_s0.png')).astype(int)).max())
                   for i, value in zip(ids, plain)]
    print('REPRODUCTION', {'old_joint_pixel_max': max(errors), 'historical_text_pixel_max': max(text_errors),
                          'fresh_original_text_raw_exact': True}, flush=True)
    assert max(text_errors) <= 1
    atomic_json(OUT/'reproduction.json', {'old_joint_cases': 32, 'max_pixel_difference': max(errors),
        'old_output_exact_cases': sum(x == 0 for x in errors), 'new_text_cases': 32,
        'new_text_max_pixel_difference': max(text_errors), 'all_backbone_digests_equal': True,
        'new_vs_fresh_original_raw_tensor_exact': True,
        'historical_note': 'Historical PNGs were generated under no_grad; fresh model evaluation is under inference_mode. Allow at most one uint8 quantization level against historical files, and require raw exactness against a fresh original under the same conditions.'})
    del original
    for row in primary:
        for seed_index in range(2):
            for source, target in [('text', 'text'), ('image', 'old_image'), ('joint', 'old_joint')]:
                shutil.copy2(SOURCE/'native'/f'{row["case_id"]}_{source}_s{seed_index}.png',
                             OUT/'native'/f'{row["case_id"]}_{target}_s{seed_index}.png')
                written += 1
    # Generate the new primary version and the same-caption validation comparisons.
    jobs = [('new', 'image', list(range(len(rows)))), ('new', 'joint', list(range(len(rows)))),
            ('old', 'image', list(range(len(primary), len(rows)))),
            ('old', 'joint', list(range(len(primary), len(rows)))),
            ('new', 'text', list(range(len(primary), len(rows))))]
    for name, mode, ids in jobs:
        for seed_index, seed in enumerate(SEEDS):
            for offset in range(0, len(ids), 32):
                selected = ids[offset:offset+32]
                values = infer(name, mode, selected, seed)
                for i, value in zip(selected, values):
                    key = 'text' if mode == 'text' else f'{name}_{mode}'
                    Image.fromarray(decode(value)).save(OUT/'native'/f'{rows[i]["case_id"]}_{key}_s{seed_index}.png')
                    written += 1
                progress('generation', model=name, mode=mode, seed=seed, done=offset+len(selected),
                         total=len(ids), images_written=written)
    last_ids = []
    for split in ('train', 'test'):
        last_ids.extend([i for i, r in enumerate(primary) if r['split'] == split][:16])
    for mode in ('image', 'joint'):
        values = infer('new_last', mode, last_ids, SEEDS[0])
        for i, value in zip(last_ids, values):
            Image.fromarray(decode(value)).save(OUT/'native'/f'{rows[i]["case_id"]}_last_{mode}_s0.png')
            written += 1
    atomic_json(OUT/'last_samples.json', [primary[i]['case_id'] for i in last_ids])
    return rows, snapshots, written


def summarize(rows, snapshots, written):
    all_metrics = []
    for row in rows:
        target = np.array(Image.open(OUT/'inputs'/f'{row["case_id"]}_target.png'))
        for seed in range(2):
            for mode in ('text', 'old_image', 'old_joint', 'new_image', 'new_joint'):
                path = OUT/'native'/f'{row["case_id"]}_{mode}_s{seed}.png'
                with Image.open(path) as im:
                    assert im.size == (16, 16) and im.mode == 'RGBA'
                    generated = np.array(im)
                all_metrics.append({'case_id': row['case_id'], 'split': row['split'], 'mode': mode,
                                    'seed': SEEDS[seed], **metrics(generated, target)})
    atomic_json(OUT/'per_sample_metrics.json', all_metrics)
    summary = {}
    for split in ('train', 'test', 'val'):
        summary[split] = {}
        for mode in ('text', 'old_image', 'old_joint', 'new_image', 'new_joint'):
            group = [r for r in all_metrics if r['split'] == split and r['mode'] == mode]
            summary[split][mode] = {key: float(np.mean([r[key] for r in group]))
                                  for key in ('premultiplied_rgba_mae', 'alpha_iou', 'foreground_pixels', 'near_empty')}
    atomic_json(OUT/'metrics_summary.json', summary)
    groups = defaultdict(list)
    for row in rows:
        if row.get('same_text_group'):
            groups[row['same_text_group']].append(row)
    response = []
    for prompt, pair in groups.items():
        for seed in range(2):
            for mode in ('text', 'old_image', 'old_joint', 'new_image', 'new_joint'):
                a, b = [np.array(Image.open(OUT/'native'/f'{r["case_id"]}_{mode}_s{seed}.png')) for r in pair]
                response.append({'prompt': prompt, 'seed': SEEDS[seed], 'mode': mode,
                                 'same_text_reference_response': metrics(a, b)['premultiplied_rgba_mae']})
    atomic_json(OUT/'same_text_reference_response.json', response)
    labels = ['HD reference', 'MC target', 'Text base', 'Old image', 'New image', 'Old joint', 'New joint']
    keys = ['reference', 'target', 'text', 'old_image', 'new_image', 'old_joint', 'new_joint']
    panels = []
    for split in ('train', 'test', 'val'):
        selected = [r for r in rows if r['split'] == split]
        for page in range((len(selected)+7)//8):
            for seed in range(2):
                name = f'{split}_comparison_{page+1:02d}_s{seed}.png'
                panel(selected[page*8:page*8+8], name, labels, keys, seed)
                panels.append(name)
        if split in ('train', 'test'):
            name = f'{split}_preview.png'
            panel(selected[:8], name, labels, keys)
    last_ids = set(json.loads((OUT/'last_samples.json').read_text()))
    for split in ('train', 'test'):
        selected = [r for r in rows if r['split'] == split and r['case_id'] in last_ids]
        for page in range(2):
            name = f'{split}_best_vs_last_{page+1:02d}.png'
            panel(selected[page*8:page*8+8], name,
                  ['HD reference', 'MC target', 'Old joint', 'New best joint', 'New last joint', 'New best image', 'New last image'],
                  ['reference', 'target', 'old_joint', 'new_joint', 'last_joint', 'new_image', 'last_image'])
            panels.append(name)
    same = [r for r in rows if r.get('same_text_group')]
    preferred = [r for r in same if r['prompt'] in ('tin helmet', 'diorite sword', 'slime sword', 'bronze hoe')]
    panel(preferred, 'same_text_preview.png', labels, keys)
    payload = json.dumps(rows, ensure_ascii=False).replace('</', '<\\/')
    gallery = '''<!doctype html><meta charset="utf-8"><title>Spatial vs pooled reference</title>
<style>body{font:15px system-ui;background:#111827;color:#e5e7eb;margin:24px}a{color:#93c5fd}select{padding:8px;margin:4px}article{border:1px solid #475569;margin:16px 0;padding:12px}.grid{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}img{width:128px;height:128px;object-fit:contain;background:#edf0f4;image-rendering:pixelated}figure:first-child img{image-rendering:auto}h2{font-size:16px}</style>
<h1>独立空间 token 与旧 MLP 适配器对比</h1><p>固定训练内64、训练未见64个目标，另含16组同文字不同设计的验证样本。双种子、32步，无结果筛选。旧版曝光更多；这是版本比较，不是相同训练预算的架构消融。</p>
<select id="split"><option value="test">训练未见</option><option value="train">训练内</option><option value="val">同文字验证组</option><option value="all">全部</option></select><select id="seed"><option value="0">种子1</option><option value="1">种子2</option></select><div id="cards"></div>
<script>const rows=DATA;const $=id=>document.getElementById(id);const esc=x=>x.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');function render(){const seed=$('seed').value;const selected=rows.filter(r=>$('split').value==='all'||r.split===$('split').value);$('cards').innerHTML=selected.map(r=>{const id=r.case_id;const cols=[['参考图',`inputs/${id}_reference.png`],['MC目标',`inputs/${id}_target.png`],...['text','old_image','new_image','old_joint','new_joint'].map((m,i)=>[['纯文本','旧版纯图','新版纯图','旧版图文','新版图文'][i],`native/${id}_${m}_s${seed}.png`])];return `<article><h2>${esc(id+' | '+r.prompt)}</h2><div class="grid">${cols.map(([label,path])=>`<figure><a href="${path}"><img loading="lazy" src="${path}"></a><figcaption>${label}</figcaption></figure>`).join('')}</div></article>`}).join('')}for(const x of ['split','seed'])$(x).onchange=render;render();</script>'''.replace('DATA', payload)
    (OUT/'index.html').write_text(gallery)
    (OUT/'README.md').write_text('# 新旧参考图方案对照\n\n训练内64、训练未见64个固定样本，以及16组同文字不同参考图（32个验证目标）。两个种子，32步Heun，原始16×16 RGBA，不做配色或轮廓修正。\n\n'
        '[可筛选图库](index.html) · [训练内预览](train_preview.png) · [训练未见预览](test_preview.png) · [同文字对照](same_text_preview.png)\n\n'
        f'旧 MLP：step {snapshots["old"]["step"]}，约 {snapshots["old"]["exposures"]:.1f} 次曝光；新方案最佳图文：step {snapshots["new"]["step"]}，约 {snapshots["new"]["exposures"]:.1f} 次曝光。预算和任务比例不同，结果不能单独归因为架构。另比较新版最后一步。\n\n'
        '训练未见样本与两种参考适配器的训练/验证目标均不重叠；它们是以前已查看过的固定测试集，本轮不用于选择检查点。旧版主要输出复用已发布评测，并复现原始批次校验；所有输入、种子、模型哈希及校验结果均有记录。\n\n'
        'metrics_summary.json 的目标误差和alpha IoU只用于辅助比较，不能代替设计保留与MC风格的人工审查。same_text_reference_response.json 仅表示换参考后输出变化的幅度，变化更大并不自动代表更好。\n\n'
        +'\n'.join(f'[{p}]({p})  ' for p in panels)+'\n')
    manifest = [{'path': str(p.relative_to(OUT)), 'sha256': file_hash(p)} for p in sorted((OUT/'native').glob('*.png'))]
    assert len(manifest) == written
    atomic_json(OUT/'output_manifest.json', manifest)
    atomic_json(OUT/'results.json', {'complete': True, 'checkpoints': snapshots, 'native_images': written,
        'primary_targets': 128, 'same_text_groups': len(groups), 'seconds': time.monotonic()-START,
        'panels': panels, 'all_output_paths_verified': True, 'no_postprocessing': True})
    if (OUT/'failed.json').exists():
        (OUT/'failed.json').rename(OUT/'first_attempt_failed.json')
    progress('complete', native_images=written)


def main():
    global OLD, NEW, SOURCE, OUT, DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--old-run', type=Path, default=OLD)
    parser.add_argument('--new-run', type=Path, default=NEW)
    parser.add_argument('--source', type=Path, default=SOURCE,
                        help='previous fixed evaluation with verified inputs, outputs and embeddings')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--memory-fraction', type=float, default=.8)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory fraction must be in (0,1]')
    OLD, NEW, SOURCE, OUT = (p.resolve() for p in (args.old_run, args.new_run, args.source, args.out))
    DEVICE = args.device
    torch.set_num_threads(4)
    torch.cuda.set_device(DEVICE)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, DEVICE)
    primary, auxiliary, pairs, lookup = prepare()
    progress('prepared', primary=len(primary), same_text=len(auxiliary)//2)
    f = features(primary, auxiliary, pairs, lookup)
    rows, snapshots, written = run_generation(primary, auxiliary, f)
    summarize(rows, snapshots, written)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            atomic_json(OUT/'failed.json', {'error': repr(exc)})
        raise
