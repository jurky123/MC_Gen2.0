"""Joint training with matched short/detail T2I, TI2I and image-only tasks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.five_task_reference import (FiveTaskData, TASK_VIEWS, accumulate_loss,
                                       configure_trainable, learning_rates, noisy_batch)
from mcgen.images import decode
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.stability_guard import StabilityGuard, StabilityPolicy
from mcgen.tasks import conditioned_prompts
from prepare_five_task_reference import DEFAULT
from prepare_reference_recovery import COMMON
from train_reference_recovery import initialize, optimizer, select_validation
from train_visual_reference import thumbnail
from validate_resolution import image_metrics


def selections(data):
    valid = select_validation(data)
    old = json.loads(Path('outputs/reference_recovery_20261108/run.json').read_text())
    train = old['training_monitor_indices'][::2]
    panels = valid[:16]+valid[24:32]+train
    controls = []
    for category in ('sword', 'axe', 'scythe', 'chestplate'):
        pool = [i for i, row in enumerate(data.rows) if row['split'] == 'val'
                and row['quality_tier'] == 'highest_quality' and data.categories[i] == category]
        assert len(pool) >= 2
        controls += [(pool[0], pool[1]), (pool[1], pool[0])]
    assert all(data.rows[i]['split'] == 'val' for i in valid)
    assert all(data.rows[i]['split'] == 'train' for i in train)
    assert len(panels) == 32
    return valid, panels, controls


def fixed_noise(n, seed, device):
    return torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(seed)).repeat(n, 1, 1, 1).to(device)


def draw_sheet(path, title, labels, entries):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    canvas = Image.new('RGB', (14+154*len(labels), 66+164*len(entries)), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, font=font, fill='white')
    for c, text in enumerate(labels):
        draw.text((14+c*154, 34), text, font=font, fill='#cbd5e1')
    for j, (paths, line1, line2) in enumerate(entries):
        y = 62+164*j
        for c, source in enumerate(paths):
            with Image.open(source) as im:
                canvas.paste(thumbnail(im, 128, native=im.size == (16, 16)), (14+c*154, y))
        draw.text((14, y+132), line1[:125], font=font, fill='white')
        draw.text((14, y+147), line2[:125], font=font, fill='#aebbd0')
    canvas.save(path)


@torch.inference_mode()
def evaluate(model, data, root, step, device, valid, panels, controls):
    directory = root/'generation_checks'/f'{step:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    score = {'step': step, 'flow': {}, 'generation': {}, 'native_size': 16,
             'main_generation_precision': 'bfloat16', 'generation_cfg': 1.}
    for name, task, variant in TASK_VIEWS:
        target, h, mask, refs, present = data.batch(valid, [task]*len(valid), [variant]*len(valid), device)
        for k, level in enumerate((.1, .5, .9, .99)):
            eps = torch.randn(target.shape, generator=torch.Generator(device=device).manual_seed(2026110901+k), device=device)
            x = target*(1-level)+eps*level
            parts = []
            for start in range(0, len(valid), 8):
                stop = min(start+8, len(valid))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ctx = model.context(h[start:stop], mask[start:stop],
                        refs[start:stop] if task != 'text' else None,
                        present[start:stop] if task != 'text' else None)
                    pred = model(x[start:stop], torch.full((stop-start,), level, device=device), ctx)
                parts += (pred.float()-(eps-target)[start:stop]).square().mean((1, 2, 3)).cpu().tolist()
            score['flow'][f'{name}_t{level}'] = {'hq': float(np.mean(parts[:24])),
                'legacy': float(np.mean(parts[24:])), 'all': float(np.mean(parts))}
        for start in range(0, len(panels), 8):
            indices = panels[start:start+8]
            _, h, mask, refs, present = data.batch(indices, [task]*len(indices), [variant]*len(indices), device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                images = sample(model, h, mask, refs if task != 'text' else None,
                    present if task != 'text' else None, seeds=[2026110907]*len(indices), steps=24,
                    text_scale=1., ref_scale=1., initial_noise=fixed_noise(len(indices), 2026110907, device))
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError('Nonfinite generation')
            for j, value in enumerate(images):
                Image.fromarray(decode(value)).save(directory/f'{start+j:02d}_{name}.png')
        metrics = []
        for j, i in enumerate(panels):
            with Image.open(data.rows[i]['target_file']) as im:
                truth = np.array(im.convert('RGBA')).repeat(2, 0).repeat(2, 1)
            metrics.append(image_metrics(directory/f'{j:02d}_{name}.png', truth))
        score['generation'][name] = {}
        for split, group in [('hq_val', metrics[:16]), ('legacy_val', metrics[16:24]), ('train', metrics[24:])]:
            score['generation'][name][split] = {k: float(np.mean([x[k] for x in group])) for k in ('premultiplied_mae', 'alpha_iou')}
    score['balanced_generation_mae'] = float(np.mean([score['generation'][n]['hq_val']['premultiplied_mae'] for n, _, _ in TASK_VIEWS]))
    score['text_generation_mae'] = float(np.mean([score['generation'][n]['hq_val']['premultiplied_mae'] for n in ('text_short', 'text_detail')]))
    score['reference_generation_mae'] = float(np.mean([score['generation'][n]['hq_val']['premultiplied_mae'] for n in ('joint_short', 'joint_detail', 'reference')]))
    for page, (name, indices) in enumerate([('val_hq_01', range(0, 8)), ('val_hq_02', range(8, 16)),
                                          ('val_legacy', range(16, 24)), ('train', range(24, 32))]):
        entries = []
        for j in indices:
            row = data.rows[panels[j]]
            paths = [row['reference_file'], row['target_file']]+[directory/f'{j:02d}_{n}.png'
                for n in ('text_short', 'text_detail', 'joint_short', 'joint_detail', 'reference')]
            entries.append((paths, f'{row["split"]} | {row["captions"][0]}',
                'Distinct verified detail' if len(row['captions']) > 1 else 'No fine annotation: detail branch uses original short label'))
        draw_sheet(directory/f'{name}.png', f'Update{step} | five tasks | fixed noise | {name}',
            ['HD reference', 'MC target', 'Short T2I', 'Detail T2I', 'Short + image', 'Detail + image', 'Image only'], entries)
    atomic_json(directory/'panel.json', [data.rows[i] for i in panels])
    # Change only the caption while keeping the HD reference and noise fixed.
    ids = [a for a, _ in controls]
    other = [b for _, b in controls]
    _, h, mask, refs, present = data.batch(ids, ['joint']*len(ids), [1]*len(ids), device)
    _, swapped, swapped_mask, _, _ = data.batch(other, ['joint']*len(ids), [1]*len(ids), device)
    control_outputs = {}
    for label, text, m, r in [('joint_a', h, mask, refs), ('joint_b', swapped, swapped_mask, refs),
                              ('text_a', h, mask, None), ('text_b', swapped, swapped_mask, None),
                              ('image', h, torch.zeros_like(mask), refs)]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values = sample(model, text, m, r, present if r is not None else None,
                seeds=[2026110911]*len(ids), steps=24, text_scale=1., ref_scale=1.,
                initial_noise=fixed_noise(len(ids), 2026110911, device))
        control_outputs[label] = values.cpu()
        for j, value in enumerate(values):
            Image.fromarray(decode(value)).save(directory/f'control_{j:02d}_{label}.png')
    entries = []
    descriptions = []
    for j, (a, b) in enumerate(controls):
        ra, rb = data.rows[a], data.rows[b]
        entries.append(([ra['reference_file'], ra['target_file']]+[directory/f'control_{j:02d}_{k}.png'
            for k in ('joint_a', 'joint_b', 'text_a', 'text_b', 'image')],
            'Text A: '+ra['captions'][0], 'Text B: '+rb['captions'][0]+' | same reference, same noise'))
        descriptions.append({'id': ra['id'], 'split': 'val', 'text_a': ra['captions'][-1],
                             'text_b': rb['captions'][-1], 'other_id': rb['id']})
    draw_sheet(directory/'text_control.png', f'Update{step} | held-out: fixed HD reference, different existing captions',
        ['Fixed reference', 'MC target A', 'Joint text A', 'Joint text B', 'T2I text A', 'T2I text B', 'Image only'], entries)
    score['caption_response_l1'] = {'joint': (control_outputs['joint_a']-control_outputs['joint_b']).abs().mean().item(),
                                   'text': (control_outputs['text_a']-control_outputs['text_b']).abs().mean().item(),
                                   'interpretation': 'Response diagnostic only; disagreement with reference has no target and is not optimized.'}
    atomic_json(directory/'text_control.json', descriptions)
    # Match the earlier FP32 common-prompt evaluation exactly.
    h, mask = data.encoder.encode(conditioned_prompts(COMMON, data.cfg), device)
    common = sample(model, h, mask, seeds=[2026110841]*len(COMMON), steps=32, text_scale=2.,
                    initial_noise=fixed_noise(len(COMMON), 2026110841, device))
    entries = []
    drifts = []
    previous = Path('outputs/reference_recovery_evaluation_20261108/native')
    for j, value in enumerate(common):
        path = directory/f'common_{j:02d}.png'
        pixels = decode(value)
        Image.fromarray(pixels).save(path)
        with Image.open(previous/f'common_{j:02d}_last_s0.png') as im:
            drifts.append(np.abs(pixels.astype(np.int16)-np.array(im).astype(np.int16)))
        entries.append(([previous/f'common_{j:02d}_before_s0.png', previous/f'common_{j:02d}_last_s0.png', path], COMMON[j], 'Pure T2I | CFG2 | FP32 | fixed prompt and noise'))
    draw_sheet(directory/'common_text.png', f'Update{step} | common T2I | FP32 regression', ['Original270', 'Previous8000', 'Current'], entries)
    score['common_text_drift_from_source'] = {'mean_uint8': float(np.mean(drifts)), 'max_uint8': int(np.max(drifts))}
    atomic_json(directory/'metrics.json', score)
    atomic_json(root/'validation_latest.json', score)
    return score


def publish(root, step):
    import html
    import shutil
    destination = Path('outputs/native16_data_review_20261107/five-task')
    (destination/'panels').mkdir(parents=True, exist_ok=True)
    body = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五任务平衡训练</title><style>body{font:16px/1.6 system-ui;background:#111827;color:#e5e7eb;max-width:1120px;margin:24px auto;padding:0 16px}a{color:#93c5fd}img{max-width:100%}details{background:#1f2937;padding:16px;margin:18px 0;border-radius:10px}summary{cursor:pointer}</style><p><a href="/recovery-evaluation/">上一轮完整评估</a> · <a href="/five-task-evaluation/">本轮训练结束后完整评估</a></p><h1>五任务平衡训练：验证集优先</h1><p>每对配对同时构造粗标/精标图文生成、粗标/精标文生图、纯参考图五种任务。粗标使用原始名称；旧配对缺精标时如实回退原标签。训练集和划分不变。新增16000次更新，有效batch160；从第一步联合训练。</p><p>对照列为高清参考、MC目标、粗标T2I、精标T2I、粗标图文、精标图文、纯参考。前两图为16个高质量验证目标，第三图为8个旧验证目标，第四图为8个训练目标。常见文生图另外与更早270步、上一轮8000步对照。</p><p>固定参考换文字只检查文字响应；文字与参考冲突时没有配对真值，不将变化幅度当作质量分数。此页不使用测试集选模型；训练完成后另做全部56个测试目标的双种子评估。</p>'''
    names = [('val_hq_01', '验证集：高质量1'), ('val_hq_02', '验证集：高质量2'),
             ('val_legacy', '验证集：旧数据'), ('train', '训练集'),
             ('text_control', '验证集：固定参考更换文字'), ('common_text', '常见文生图回归')]
    for directory in sorted((root/'generation_checks').iterdir(), reverse=True):
        if not (directory/'metrics.json').exists():
            continue
        scores = json.loads((directory/'metrics.json').read_text())
        current = scores['step']
        body += f'<details {"open" if current == step else ""}><summary>更新{current}/16000 · 五任务验证MAE {scores["balanced_generation_mae"]:.4f} · 文生图MAE {scores["text_generation_mae"]:.4f}</summary>'
        for stem, label in names:
            filename = f'{directory.name}_{stem}.png'
            shutil.copy2(directory/f'{stem}.png', destination/'panels'/filename)
            body += f'<h3>{label}</h3><a href="panels/{filename}">原图</a><br><img loading="lazy" src="panels/{filename}">'
        body += '<details><summary>文字响应对照：实际精标文本</summary>'
        for row in json.loads((directory/'text_control.json').read_text()):
            body += '<p>A: '+html.escape(row['text_a'])+'<br>B: '+html.escape(row['text_b'])+'</p>'
        body += '</details></details>'
    path = destination/'index.html'
    temporary = path.with_suffix('.html.tmp')
    temporary.write_text(body)
    temporary.replace(path)


def train(root, device):
    status_path = root/'status.json'
    if status_path.exists():
        previous_status = json.loads(status_path.read_text())
        if previous_status.get('stop_reason') == 'stability_guard':
            raise ValueError('This run halted for divergence; start a new experiment from a validated checkpoint')
        if previous_status.get('complete'):
            print('ALREADY_COMPLETE: preserving existing checkpoints', flush=True)
            return
    print('LOAD_DATA', flush=True)
    data = FiveTaskData(root)
    data.cache_device(device)
    recipe = json.loads((root/'recipe.json').read_text())
    contract = file_hash(root/'recipe.json')
    checks = json.loads((root/'checks.json').read_text())
    if not checks['passed']:
        raise ValueError('GPU checks must pass')
    micro = checks['chosen']['microbatch']
    print('INITIALIZE_MODEL', 'microbatch', micro, flush=True)
    model, info = initialize(data, device)
    freeze_time = recipe.get('freeze_time_mlp', False)
    configure_trainable(model, freeze_time)
    initial_time = {k: v.detach().clone() for k, v in model.time.state_dict().items()}
    opt = optimizer(model, recipe)
    valid, panels, controls = selections(data)
    directory = root/'training'
    directory.mkdir(exist_ok=True)
    step = 0
    stop_reason = None
    guard = StabilityGuard(StabilityPolicy(**recipe.get('stability_guard',
        {'warmup': recipe['warmup_steps'], 'gradient_clip': recipe['gradient_clip']})))
    best = {'balanced': float('inf'), 'text': float('inf'), 'reference': float('inf')}
    counts = Counter()
    if (directory/'latest.pt').exists():
        state = load_checkpoint(directory/'latest.pt')
        if state['recipe_sha256'] != contract or state['catalog_fingerprint'] != data.pairs_sha256 or state['config'] != data.cfg.to_dict():
            raise ValueError('Resume contract changed')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        step, best, counts = state['step'], state['best_generation'], Counter(state['task_counts'])
        if 'stability_guard' in state:
            guard = StabilityGuard.from_state(state['stability_guard'])
    atomic_json(root/'run.json', {'recipe': recipe, 'recipe_sha256': contract, 'config': data.cfg.to_dict(),
        'pairs_sha256': data.pairs_sha256, 'initialization': info, 'microbatch': micro,
        'validation_indices': valid, 'generation_panel_indices': panels, 'text_control_pairs': controls,
        'test_used_for_selection': False, 'all_tasks_train_from_first_step': True})
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))

    def save(name):
        save_checkpoint(directory/name, {'config': data.cfg.to_dict(), 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'step': step, 'best_generation': best, 'task_counts': dict(counts),
            'recipe_sha256': contract, 'catalog_fingerprint': data.pairs_sha256,
            'text_identity': data.encoder.identity, 'visual_identity': data.visual_identity,
            'initialization': info, 'training_protocol': recipe['version'],
            'stability_guard': guard.state_dict(), 'stability_halted': stop_reason == 'stability_guard',
            'diagnostic_only': stop_reason == 'stability_guard'})

    validation_seconds = 0.
    validation_calls = 0
    train_seconds = 0.
    times = []
    started = time.monotonic()

    def validate():
        nonlocal validation_seconds, validation_calls
        begin = time.monotonic()
        scores = evaluate(model, data, root, step, device, valid, panels, controls)
        if freeze_time:
            if not all(torch.equal(initial_time[k], v) for k, v in model.time.state_dict().items()):
                raise RuntimeError('Frozen timestep MLP changed')
            scores['frozen_time_mlp_bitwise_unchanged'] = True
            atomic_json(root/'validation_latest.json', scores)
            atomic_json(root/'generation_checks'/f'{step:06d}'/'metrics.json', scores)
        for key in best:
            metric = scores[key+'_generation_mae']
            if metric < best[key]:
                best[key] = metric
                save('best_'+key+'.pt')
        save('latest.pt')
        with (directory/'validation.jsonl').open('a') as f:
            f.write(json.dumps(scores)+'\n')
        publish(root, step)
        validation_seconds += time.monotonic()-begin
        validation_calls += 1
        print('VALIDATION', step, 'balanced', scores['balanced_generation_mae'],
              'text', scores['text_generation_mae'], 'reference', scores['reference_generation_mae'], flush=True)

    if step == 0:
        print('INITIAL_VALIDATION', flush=True)
        validate()
    for update in range(step, recipe['steps']):
        if stopping[0]:
            break
        begin = time.monotonic()
        rng = torch.Generator().manual_seed(data.cfg.train.seed+1000003*update)
        plan = data.five_sampler.sample(recipe['pairs_per_update'], recipe['unconditional_per_update'], rng)
        batch = data.plan_batch(plan, device)
        noisy = noisy_batch(batch[0], data.cfg.train.seed+104729*(update+1), device)
        for group, rate in zip(opt.param_groups, learning_rates(update, recipe)):
            group['lr'] = rate
        opt.zero_grad(set_to_none=True)
        losses = accumulate_loss(model, batch, noisy, micro)
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError('Nonfinite training loss')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True)
        health = guard.observe(update+1, losses.mean().item(), norm.item())
        if health is not None:
            atomic_json(root/'stability_latest.json', health)
            if health['stop']:
                stop_reason = 'stability_guard'
                atomic_json(root/'stability_stop.json', {**health, 'last_optimizer_update': step,
                    'recommended_checkpoint': str((directory/'best_balanced.pt').resolve()),
                    'action': 'skip proposed optimizer step; preserve best and latest snapshots; save diagnostic halted.pt'})
                opt.zero_grad(set_to_none=True)
                print('STABILITY_HALT', json.dumps(health), flush=True)
                break
        opt.step()
        counts.update(plan.names)
        step = update+1
        duration = time.monotonic()-begin
        times.append(duration)
        times = times[-100:]
        train_seconds += duration
        metric = {'step': step, 'loss': losses.mean().item(), 'gradient_norm': norm.item(),
                  'base_lr': opt.param_groups[0]['lr'], 'visual_lr': opt.param_groups[1]['lr'], 'task_counts': dict(counts)}
        for name in sorted(set(plan.names)):
            metric['loss_'+name] = losses[[j for j, n in enumerate(plan.names) if n == name]].mean().item()
        with (directory/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric)+'\n')
        remaining_checks = sum(x > step for x in recipe['evaluate_updates'])
        eta = (recipe['steps']-step)*float(np.median(times))+remaining_checks*validation_seconds/max(1, validation_calls)
        if step % 10 == 0 or step == 1:
            atomic_json(root/'status.json', {**metric, 'complete': False, 'planned_steps': recipe['steps'],
                'microbatch': micro, 'logical_batch': 160, 'elapsed_seconds': time.monotonic()-started,
                'training_seconds': train_seconds, 'estimated_remaining_seconds': eta,
                'recent_seconds_per_update': float(np.median(times))})
        if step % 100 == 0:
            print('STEP', step, 'loss', round(metric['loss'], 6), 'seconds/update', round(float(np.median(times)), 3), flush=True)
        if step in recipe['evaluate_updates']:
            validate()
        elif step % 500 == 0:
            save('latest.pt')
    save('halted.pt' if stop_reason == 'stability_guard' else 'latest.pt')
    atomic_json(root/'status.json', {'step': step, 'planned_steps': recipe['steps'], 'complete': step == recipe['steps'],
        'stopped': stopping[0] or stop_reason is not None, 'stop_reason': stop_reason,
        'elapsed_seconds': time.monotonic()-started, 'best_generation': best,
        'task_counts': dict(counts), 'initial_checkpoint_unchanged': file_hash(info['checkpoint']) == info['sha256']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.5, args.device)
    train(args.out, args.device)
