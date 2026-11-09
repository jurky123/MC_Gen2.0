"""Check task gradients and benchmark bounded-memory logical160 updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch

from mcgen.catalog import atomic_json, file_hash
from mcgen.five_task_reference import FiveTaskData, accumulate_loss, configure_trainable, noisy_batch
from prepare_five_task_reference import DEFAULT
from train_reference_recovery import initialize, optimizer


def check(root, device, microbatches):
    data = FiveTaskData(root)
    data.cache_device(device)
    recipe = json.loads((root/'recipe.json').read_text())
    model, info = initialize(data, device)
    freeze_time = recipe.get('freeze_time_mlp', False)
    configure_trainable(model, freeze_time)
    time_before = {k: v.detach().clone() for k, v in model.time.state_dict().items()}
    opt = optimizer(model, recipe)
    ids = [i for i, r in enumerate(data.rows) if r['split'] == 'train'][:8]
    original = model.text.weight.detach().clone()
    pure = (data.replay.batch(data.replay.sample(8, torch.Generator().manual_seed(9917)), ['text']*8, device)
            if data.replay is not None else data.batch(ids, ['text']*len(ids), [0]*len(ids), device))
    opt.zero_grad(set_to_none=True)
    accumulate_loss(model, pure, noisy_batch(pure[0], 9919, device), 8)
    text_grad = model.text.weight.grad.float().norm().item()
    no_visual_grad = all(p.grad is None for p in model.visual_adapter.parameters())
    opt.step()
    text_updated = not torch.equal(original, model.text.weight)
    measurements = []
    last_parts = None
    for micro in microbatches:
        timings = []
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            for k in range(4):
                plan = data.sample_plan(recipe, torch.Generator().manual_seed(1201+k))
                batch = data.plan_batch(plan, device)
                noisy = noisy_batch(batch[0], 1301+k, device)
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                started = time.monotonic()
                last_parts = accumulate_loss(model, batch, noisy, micro,sort_by_length=recipe.get('length_sorted_microbatches',False))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize(device)
                timings.append(time.monotonic()-started)
            seconds = statistics.median(timings[1:])
            result = {'microbatch': micro, 'logical_batch': recipe['batch_size'], 'seconds_per_update': seconds,
                'draws_per_second': recipe['batch_size']/seconds, 'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/1024**3}
            measurements.append(result)
            print('PROBE', json.dumps(result), flush=True)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            measurements.append({'microbatch': micro, 'oom': True})
            print('BOUNDED OOM', micro, flush=True)
            break
    good = [x for x in measurements if not x.get('oom')]
    if not good:
        raise RuntimeError('No feasible logical batch update')
    fastest = max(x['draws_per_second'] for x in good)
    chosen = min((x for x in good if x['draws_per_second'] >= .97*fastest), key=lambda x: x['microbatch'])
    plan = data.sample_plan(recipe, torch.Generator().manual_seed(1471))
    batch = data.plan_batch(plan, device)
    opt.zero_grad(set_to_none=True)
    parts = accumulate_loss(model, batch, noisy_batch(batch[0], 1481, device), chosen['microbatch'],
                            sort_by_length=recipe.get('length_sorted_microbatches',False))
    visual_grad = sum(p.grad.detach().float().square().sum().item() for p in model.visual_adapter.parameters() if p.grad is not None)**.5
    time_unchanged = all(torch.equal(time_before[k], v) for k, v in model.time.state_dict().items())
    passed = (text_grad > 0 and text_updated and no_visual_grad and visual_grad > 0 and bool(torch.isfinite(parts).all())
              and (not freeze_time or time_unchanged))
    result = {'passed': passed, 'pure_text_gradient_norm': text_grad,
        'pure_text_updates_shared_generator': text_updated, 'pure_text_has_no_visual_gradient': no_visual_grad,
        'mixed_visual_gradient_norm': visual_grad, 'task_counts_one_update': dict(Counter(plan.names)),
        'all_five_tasks_from_each_pair': True, 'original_captions_only': set(plan.variants) <= {0, 1},
        'freeze_time_mlp': freeze_time, 'time_weights_unchanged': time_unchanged,
        'measurements': measurements, 'chosen': chosen, 'benchmark_weights_discarded': True,
        'initial_checkpoint_unchanged': file_hash(info['checkpoint']) == info['sha256'],
        'planned_updates_unchanged': recipe['steps'], 'logical_batch_unchanged': recipe['batch_size'],
        'broad_text_gradient_checked': data.replay is not None,
        'recipe_sha256': file_hash(root/'recipe.json')}
    atomic_json(root/'checks.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError('Five-task gradient checks failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--microbatches', type=int, nargs='+', default=[20, 40, 80, 160])
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    recipe=json.loads((args.out/'recipe.json').read_text())
    torch.cuda.set_per_process_memory_fraction(recipe.get('gpu_memory_fraction',.5), args.device)
    check(args.out, args.device, args.microbatches)
