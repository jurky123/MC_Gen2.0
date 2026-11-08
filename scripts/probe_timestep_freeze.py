"""Matched continuation experiment: original training versus frozen time MLP."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData, accumulate_loss, learning_rates, noisy_batch
from mcgen.images import decode
from mcgen.model import PixelFlow
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.stability_guard import StabilityGuard
from mcgen.tasks import conditioned_prompts
from prepare_reference_recovery import COMMON
from train_five_task_reference import draw_sheet, fixed_noise
from train_reference_recovery import optimizer
from validate_resolution import image_metrics

ROOT = Path('outputs/reference_five_task_20261108')
DEFAULT = Path('outputs/five_task_instability_20261108/continuation')


def time_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.time.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def evaluate(model, data, directory, step, device, panel):
    destination = directory/f'eval_{step:06d}'
    destination.mkdir(exist_ok=True)
    h, mask = data.encoder.encode(conditioned_prompts(COMMON, data.cfg), device)
    values = sample(model, h, mask, seeds=[2026110841]*len(COMMON), steps=32, text_scale=2.,
                    initial_noise=fixed_noise(len(COMMON), 2026110841, device))
    for j, value in enumerate(values):
        Image.fromarray(decode(value)).save(destination/f'common_{j:02d}.png')
    summary = {}
    for name, task, variant in [('joint', 'joint', 1), ('text', 'text', 1)]:
        metrics = []
        for start in range(0, len(panel), 8):
            ids = panel[start:start+8]
            target, h, mask, refs, present = data.batch(ids, [task]*len(ids), [variant]*len(ids), device)
            result = sample(model, h, mask, refs if task == 'joint' else None, present if task == 'joint' else None,
                seeds=[2026110907]*len(ids), steps=24, text_scale=1., ref_scale=1.,
                initial_noise=fixed_noise(len(ids), 2026110907, device))
            for j, value in enumerate(result):
                path = destination/f'{start+j:02d}_{name}.png'
                Image.fromarray(decode(value)).save(path)
                with Image.open(data.rows[ids[j]]['target_file']) as im:
                    truth = np.array(im.convert('RGBA')).repeat(2, 0).repeat(2, 1)
                metrics.append(image_metrics(path, truth))
        summary[name] = {split: {key: float(np.mean([x[key] for x in group])) for key in ('premultiplied_mae', 'alpha_iou')}
                        for split, group in [('val', metrics[:8]), ('train', metrics[8:])]}
    entries = []
    for j, prompt in enumerate(COMMON):
        baseline = Path('outputs/reference_five_task_20261108/generation_checks/008000')/f'common_{j:02d}.png'
        entries.append(([baseline, destination/f'common_{j:02d}.png'], prompt, 'FP32 matched prompt and Gaussian noise'))
    draw_sheet(destination/'common.png', f'{directory.name} | continued step{step}', ['Initial healthy8000', 'Current'], entries)
    entries = []
    for j, i in enumerate(panel):
        row = data.rows[i]
        entries.append(([row['reference_file'], row['target_file'], destination/f'{j:02d}_text.png', destination/f'{j:02d}_joint.png'],
                        row['split']+' | '+row['captions'][0], 'Same caption and noise; diagnostic continuation'))
    draw_sheet(destination/'paired.png', f'{directory.name} | step{step} | held-out first8, train last8',
               ['HD reference', 'MC target', 'Detail T2I', 'Detail + image'], entries)
    atomic_json(destination/'metrics.json', summary)
    return summary


def train(root, arm, device, steps):
    directory = root/arm
    directory.mkdir(parents=True, exist_ok=False)
    data = FiveTaskData(ROOT)
    data.cache_device(device)
    recipe = json.loads((ROOT/'recipe.json').read_text())
    path = ROOT/'training/best_text.pt'
    digest = file_hash(path)
    state = load_checkpoint(path)
    assert state['step'] == 8000
    cfg = Experiment.from_dict(state['config'])
    model = PixelFlow(cfg.model).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    opt = optimizer(model, recipe)
    opt.load_state_dict(state['optimizer'])
    if arm == 'freeze_time':
        model.time.requires_grad_(False)
    original_time = time_digest(model)
    del state
    gc.collect()
    run = json.loads((ROOT/'run.json').read_text())
    panel = run['validation_indices'][:8]+run['generation_panel_indices'][-8:]
    guard = StabilityGuard()
    for line in (ROOT/'training/metrics.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item['step'] > 8000:
            break
        guard.observe(item['step'], item['loss'], item['gradient_norm'])
    atomic_json(directory/'protocol.json', {'arm': arm, 'source': str(path.resolve()), 'source_sha256': digest,
        'source_step': 8000, 'steps': steps, 'optimizer_restored': True, 'only_difference': 'time MLP requires_grad',
        'recipe': recipe, 'diagnostic_only': True, 'production_checkpoint_overwritten': False,
        'panel_indices': panel, 'test_used': False, 'time_digest_before': original_time})
    print('INITIAL_EVAL', arm, flush=True)
    evaluate(model, data, directory, 8000, device, panel)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    started = time.monotonic()
    step = 8000
    halt = None
    for update in range(8000, 8000+steps):
        if stopping[0]:
            break
        plan = data.five_sampler.sample(31, 5, torch.Generator().manual_seed(cfg.train.seed+1000003*update))
        batch = data.plan_batch(plan, device)
        noisy = noisy_batch(batch[0], cfg.train.seed+104729*(update+1), device)
        for group, rate in zip(opt.param_groups, learning_rates(update, recipe)):
            group['lr'] = rate
        opt.zero_grad(set_to_none=True)
        losses = accumulate_loss(model, batch, noisy, 160)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        loss, norm_value = losses.mean().item(), norm.item()
        decision = guard.observe(update+1, loss, norm_value)
        if decision is not None and decision['stop']:
            halt = decision
            print('GUARD_HALT', arm, decision, flush=True)
            break
        opt.step()
        step = update+1
        item = {'step': step, 'loss': loss, 'gradient_norm': norm_value, 'seconds': time.monotonic()-started}
        with (directory/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(item)+'\n')
        if step % 100 == 0:
            atomic_json(directory/'status.json', item)
            print('STEP', arm, step, round(loss, 5), round(norm_value, 3), flush=True)
        if step % 1000 == 0:
            scores = evaluate(model, data, directory, step, device, panel)
            print('EVAL', arm, step, scores, flush=True)
    if step % 1000 or halt:
        evaluate(model, data, directory, step, device, panel)
    after_time = time_digest(model)
    if arm == 'freeze_time':
        assert after_time == original_time
    save_checkpoint(directory/'diagnostic.pt', {'config': cfg.to_dict(), 'model': model.state_dict(),
        'step': step, 'diagnostic_only': True, 'target_leakage': False,
        'text_identity': data.encoder.identity, 'visual_identity': data.visual_identity,
        'source_sha256': digest, 'arm': arm})
    assert file_hash(path) == digest
    atomic_json(directory/'status.json', {'complete': step == 8000+steps, 'step': step, 'halt': halt,
        'seconds': time.monotonic()-started, 'time_weights_bitwise_unchanged': original_time == after_time,
        'source_checkpoint_unchanged': True, 'diagnostic_only': True})
    print('COMPLETE', arm, step, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--arm', choices=('control', 'freeze_time'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=4000)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.45, args.device)
    train(args.out, args.arm, args.device, args.steps)
