"""HQ reference/text fine-tuning at the prepared native target resolution."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import signal
import statistics
import time

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.curated_training import (MixedData, initialize_mixed,
                                   mixed_training_budget, mixed_validation_ids)
from mcgen.images import decode
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from train_visual_reference import thumbnail


def initialize(data, device):
    info = json.loads((data.root/'initialization.json').read_text())
    if file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Initialization checkpoint changed')
    state = load_checkpoint(info['checkpoint'])
    if state['text_identity'] != data.encoder.identity or state['visual_identity'] != data.visual_identity:
        raise ValueError('Frozen encoder identity changed')
    torch.manual_seed(data.cfg.train.seed)
    return initialize_mixed(data.cfg, state, device), info


def optimizer(model, lr):
    base = [p for name, p in model.named_parameters() if not name.startswith('visual_adapter.')]
    return torch.optim.AdamW([{'params': base, 'lr': lr, 'lr_factor': 1.},
        {'params': model.visual_adapter.parameters(), 'lr': lr*10, 'lr_factor': 10.}], weight_decay=.01)


def loss_for(model, batch, device, seed):
    target, text, mask, refs, present = batch
    rng = torch.Generator(device=device).manual_seed(seed)
    t = torch.rand(len(target), generator=rng, device=device)
    noise = torch.randn(target.shape, generator=rng, device=device)
    noisy = target*(1-t[:, None, None, None])+noise*t[:, None, None, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(noisy, t, model.context(text, mask, refs, present))
        per_sample = (pred.float()-(noise-target)).square().mean((1, 2, 3))
    return per_sample.mean(), per_sample


def smoke(root, device):
    data = MixedData(root)
    model, info = initialize(data, device)
    ids = [i for i, row in enumerate(data.rows) if row['split'] == 'train' and row['quality_tier'] == 'highest_quality'][:16]
    tasks = ['text']*8+['joint']*8
    batch = data.batch(ids, tasks, [1]*len(ids), device)
    with torch.no_grad():
        before = loss_for(model, batch, device, 7919)[0].item()
    opt = optimizer(model, 3e-5)
    original = model.text.weight.detach().clone()
    # A pure-text-only update must reach shared generator parameters.
    pure = data.batch(ids[:8], ['text']*8, [1]*8, device)
    opt.zero_grad(set_to_none=True)
    loss_for(model, pure, device, 9011)[0].backward()
    text_grad = model.text.weight.grad.norm().item()
    visual_grad_absent = model.visual_adapter.projection[-1].weight.grad is None
    opt.step()
    text_updated = not torch.equal(original, model.text.weight)
    for update in range(60):
        opt.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, device, 10007+update)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if update%20 == 19:
            print('SMOKE', update+1, loss.item(), flush=True)
    with torch.no_grad():
        after = loss_for(model, batch, device, 7919)[0].item()
    report = {'passed': after < before and text_grad > 0 and text_updated and visual_grad_absent,
        'before': before, 'after': after, 'text_only_gradient_norm': text_grad,
        'text_only_updates_shared_parameters': text_updated, 'text_only_bypasses_visual_adapter': visual_grad_absent,
        'target_native_size': data.cfg.model.size, 'weights_reused': False,
        'initial_checkpoint_unchanged': file_hash(info['checkpoint']) == info['sha256']}
    atomic_json(root/'smoke.json', report)
    print('SMOKE RESULT', report, flush=True)
    if not report['passed']:
        raise RuntimeError('Mixed-task learning check failed')


def probe(root, device, memory_fraction=.38):
    data = MixedData(root)
    model, _ = initialize(data, device)
    opt = optimizer(model, data.cfg.train.lr)
    results = []
    free, total = torch.cuda.mem_get_info(device)
    candidates = ((32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
                  if data.cfg.model.size == 16 else (8, 16, 24, 32, 48, 64, 96, 128))
    for n in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for k in range(4):
                ids, tasks, variants = data.sampler.sample(n, torch.Generator().manual_seed(k))
                batch = data.batch(ids, tasks, variants, device)
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                started = time.monotonic()
                loss = loss_for(model, batch, device, 104729+k)[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize(device)
                times.append(time.monotonic()-started)
            seconds = statistics.median(times[1:])
            row = {'batch_size': n, 'seconds_per_step': seconds, 'images_per_second': n/seconds,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/1024**3}
            results.append(row)
            print('BATCH', row, flush=True)
            del batch, loss
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            results.append({'batch_size': n, 'oom': True})
            break
    valid = [r for r in results if not r.get('oom')]
    if not valid:
        raise RuntimeError('No feasible mixed-training batch')
    fastest = max(r['images_per_second'] for r in valid)
    chosen = min((r for r in valid if r['images_per_second'] >= fastest*.97), key=lambda r: r['batch_size'])
    cfg = data.cfg
    targets = len({r['target_hash'] for r in data.rows if r['split'] == 'train'})
    cfg.train.batch_size = chosen['batch_size']
    info = json.loads((root/'initialization.json').read_text())
    draws = info.get('planned_draws_per_target', 120)
    cfg.train.steps = mixed_training_budget(targets, cfg.train.batch_size, draws)
    cfg.train.warmup = min(cfg.train.warmup, max(1, cfg.train.steps//10))
    cfg.train.eval_every = math.ceil(targets*30/cfg.train.batch_size)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    atomic_json(root/'batch_budget.json', {'measurements': results, 'chosen': chosen,
        'free_gib_before': free/1024**3, 'total_gib': total/1024**3, 'memory_limit_fraction': memory_fraction,
        'steps': cfg.train.steps, 'total_draws': cfg.train.steps*cfg.train.batch_size,
        'average_draws_per_unique_train_target': cfg.train.steps*cfg.train.batch_size/targets,
        'estimated_training_seconds': cfg.train.steps*chosen['seconds_per_step'],
        'note': f'Native{cfg.model.size} full generator backward. Encoders cached; probe weights discarded.'})


def panel_ids(data):
    result = []
    for quality, split, count in [('highest_quality', 'train', 4), ('highest_quality', 'val', 4),
                                  ('reviewed', 'train', 2), ('evaluation_only', 'val', 2)]:
        candidates = [i for i, r in enumerate(data.rows) if r['quality_tier'] == quality and r['split'] == split]
        groups = {}
        for i in candidates:
            groups.setdefault(data.rows[i]['source_family'], []).append(i)
        used = set()
        while groups and len(used) < count:
            for key in list(groups):
                pool = groups[key]
                if pool:
                    i = pool.pop(0)
                    used.add(i)
                    result.append(i)
                    if len(used) == count:
                        break
                else:
                    del groups[key]
    return result


@torch.no_grad()
def render(model, data, root, step, device, ids):
    directory = root/'generation_checks'/f'{step:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in ('text', 'reference', 'joint'):
        blocks = []
        for start in range(0, len(ids), 2):
            selected = ids[start:start+2]
            _, text, mask, refs, present = data.batch(selected, [task]*len(selected), [1]*len(selected), device)
            blocks.append(sample(model, text, mask, refs if task != 'text' else None,
                present if task != 'text' else None, seeds=[20261107]*len(selected), steps=24).cpu())
        results[task] = torch.cat(blocks)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    sheet = Image.new('RGB', (640, 55+140*len(ids)), '#111827')
    painter = ImageDraw.Draw(sheet)
    for c, name in enumerate(('HD reference', f'MC{data.cfg.model.size} target', 'Text only', 'Reference only', 'Text + reference')):
        painter.text((12+c*126, 15), name, fill='white', font=font)
    for j, i in enumerate(ids):
        row = data.rows[i]
        y = 43+j*140
        with Image.open(row['reference_file']) as im:
            pictures = [thumbnail(im)]
        pictures.append(thumbnail(Image.fromarray(decode(data.targets[i])), native=True))
        for task in ('text', 'reference', 'joint'):
            im = Image.fromarray(decode(results[task][j]))
            im.save(directory/f'{j:02d}_{task}.png')
            pictures.append(thumbnail(im, native=True))
        for c, im in enumerate(pictures):
            sheet.paste(im, (12+c*126, y))
        painter.text((12, y+115), f'{row["quality_tier"]} / {row["split"]}: {row["captions"][0]}'[:94], fill='white', font=font)
    sheet.save(directory/'comparison.png')
    atomic_json(directory/'panel.json', [data.rows[i] for i in ids])
    diversity = []
    for category in ('helmet', 'sword', 'shield', 'axe'):
        import re
        selected = [i for i, r in enumerate(data.rows) if r['quality_tier'] == 'highest_quality'
                    and r['split'] == 'val' and re.search(r'\b'+category+r'\b', r['captions'][0], re.I)][:3]
        for seed in (20261107, 20261108):
            for i in selected:
                _, _, _, refs, present = data.batch([i], ['joint'], [0], device)
                text, mask = data.encoder.encode(conditioned_prompts([category], data.cfg), device)
                result = sample(model, text, mask, refs, present, seeds=[seed], steps=24)[0]
                name = f'fixed_{category}_{i}_seed{seed}.png'
                Image.fromarray(decode(result)).save(directory/name)
                diversity.append({'prompt': category, 'seed': seed, 'output': name,
                                  'reference_file': data.rows[i]['reference_file'], 'target_hash': data.rows[i]['target_hash']})
    atomic_json(directory/'fixed_text_different_references.json', diversity)


def train(root, device, stop_after=None):
    data = MixedData(root)
    cfg = data.cfg
    model, initialization = initialize(data, device)
    opt = optimizer(model, cfg.train.lr)
    out = root/'training'
    out.mkdir(exist_ok=True)
    step, best, best_text, counts = 0, float('inf'), float('inf'), Counter()
    if (out/'latest.pt').exists():
        state = load_checkpoint(out/'latest.pt')
        if (state['config'] != cfg.to_dict() or state['catalog_fingerprint'] != data.pairs_sha256
                or state['visual_identity'] != data.visual_identity or state['text_identity'] != data.encoder.identity):
            raise ValueError('Resume data/model contract changed')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        step, best, best_text = state['step'], state['best_joint'], state['best_text']
        counts = Counter(state.get('task_counts', {}))
    start_step = step
    started = time.monotonic()
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    ids = panel_ids(data)
    val = mixed_validation_ids(data.rows, cfg.train.val_samples)
    atomic_json(root/'run.json', {'config': cfg.to_dict(), 'initialization': initialization,
        'pairs_sha256': data.pairs_sha256, 'visual_identity': data.visual_identity,
        'text_identity': data.encoder.identity, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'target_resolution': cfg.model.size, 'task_mix': cfg.train.task_mix, 'validation_indices': val,
        'panel_indices': ids, 'encoders_frozen': True, 'original_checkpoint_overwritten': False})

    def save(name):
        save_checkpoint(out/name, {'config': cfg.to_dict(), 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'step': step, 'best_joint': best, 'best_text': best_text,
            'task_counts': dict(counts), 'catalog_fingerprint': data.pairs_sha256,
            'visual_identity': data.visual_identity, 'text_identity': data.encoder.identity,
            'initialization': initialization, 'training_protocol': f'curated_native{cfg.model.size}_mixed_v1'})

    @torch.no_grad()
    def validate():
        nonlocal best, best_text
        scores = {'step': step, 'native_size': cfg.model.size}
        for mode in ('joint', 'reference', 'text'):
            target, h, mask, refs, present = data.batch(val, [mode]*len(val), [1]*len(val), device)
            losses = []
            for k, level in enumerate((.1, .5, .9)):
                rng = torch.Generator(device=device).manual_seed(104729+k)
                noise = torch.randn(target.shape, generator=rng, device=device)
                noisy = target*(1-level)+noise*level
                t = torch.full((len(val),), level, device=device)
                parts = {}
                controls = [('matched', refs)] if mode == 'text' else [('matched', refs), ('wrong_reference_same_text', refs.roll(1, 0))]
                for label, reference in controls:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(noisy, t, model.context(h, mask, reference, present))
                    parts[label] = F.mse_loss(pred.float(), noise-target).item()
                scores[f'{mode}_t{level}'] = parts
                losses.append(parts['matched'])
            scores[mode+'_flow_mse'] = sum(losses)/len(losses)
        render(model, data, root, step, device, ids)
        if scores['joint_flow_mse'] < best:
            best = scores['joint_flow_mse']
            save('best_joint_flow.pt')
        if scores['text_flow_mse'] < best_text:
            best_text = scores['text_flow_mse']
            save('best_text_flow.pt')
        save('latest.pt')
        with (out/'validation.jsonl').open('a') as f:
            f.write(json.dumps(scores)+'\n')
        atomic_json(root/'validation_latest.json', scores)
        print('VALIDATION', json.dumps(scores), flush=True)

    if step == 0:
        save(f'initial_native{cfg.model.size}.pt')
        validate()
    target_count = len({r['target_hash'] for r in data.rows if r['split'] == 'train'})
    milestones = {min(cfg.train.steps, math.ceil(target_count*e/cfg.train.batch_size)): e for e in (10, 30, 60, 90, 120)}
    end = min(cfg.train.steps, stop_after) if stop_after else cfg.train.steps
    print('TRAINING', step, end, 'batch', cfg.train.batch_size, flush=True)
    for update in range(step, end):
        if stopping[0]:
            break
        rng = torch.Generator().manual_seed(cfg.train.seed+1000003*update)
        selected, tasks, variants = data.sampler.sample(cfg.train.batch_size, rng)
        batch = data.batch(selected, tasks, variants, device)
        warm = min(1., (update+1)/cfg.train.warmup)
        progress = max(0., (update-cfg.train.warmup)/max(1, cfg.train.steps-cfg.train.warmup))
        lr = cfg.train.lr*warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:
            group['lr'] = lr*group['lr_factor']
        opt.zero_grad(set_to_none=True)
        loss, per_sample = loss_for(model, batch, device, cfg.train.seed+update+17)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite mixed-task loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        counts.update(tasks)
        step = update+1
        metric = {'step': step, 'loss': loss.item(), 'gradient_norm': norm.item(), 'lr': lr,
                  'seconds': time.monotonic()-started, 'task_counts': dict(counts)}
        detached = per_sample.detach().cpu()
        for task in set(tasks):
            positions = [j for j, name in enumerate(tasks) if name == task]
            metric['loss_'+task] = detached[positions].mean().item()
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric)+'\n')
        atomic_json(root/'status.json', {**metric, 'planned_steps': cfg.train.steps, 'complete': False,
            'average_draws_per_target': step*cfg.train.batch_size/target_count,
            'estimated_remaining_seconds': metric['seconds']/max(1, step-start_step)*(cfg.train.steps-step)})
        if step%20 == 0:
            print('STEP', step, 'loss', round(loss.item(), 6), flush=True)
        if step in milestones or step == end:
            validate()
            if step in milestones:
                save(f'exposure{milestones[step]:03d}.pt')
        elif step%100 == 0:
            save('latest.pt')
    save('latest.pt')
    atomic_json(root/'status.json', {'step': step, 'planned_steps': cfg.train.steps,
        'complete': step == cfg.train.steps, 'stopped': stopping[0], 'seconds': time.monotonic()-started,
        'best_joint': best, 'best_text': best_text, 'task_counts': dict(counts),
        'initial_checkpoint_unchanged': file_hash(initialization['checkpoint']) == initialization['sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['smoke', 'probe', 'train'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--memory-fraction', type=float, default=.38)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(a.device)
    if not 0 < a.memory_fraction <= 1:
        p.error('--memory-fraction must be between0 and1')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, a.device)
    if a.stage == 'smoke':
        smoke(a.out, a.device)
    elif a.stage == 'probe':
        probe(a.out, a.device, a.memory_fraction)
    else:
        train(a.out, a.device, a.stop_after)
