"""Train independent spatial-reference attention against an immutable MC base."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F
import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.disk_text import DiskTextEncoder
from mcgen.images import content_hash, decode, encode
from mcgen.model import PixelFlow
from mcgen.reference_adapter import initialize_visual_adapter, backbone_digest
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.tasks import conditioned_prompts
from mcgen.pair_selection import selected_indices

TEXT_CACHE = 'outputs/qwen_vl_embedding_large_20261103/features'


def prepare(root):
    audit = json.loads((root/'data_audit.json').read_text())
    base = load_checkpoint(audit['base_checkpoint'])
    if file_hash(audit['base_checkpoint']) != audit['base_checkpoint_sha256']:
        raise ValueError('Base checkpoint changed')
    data = base['config']
    data['model'].update(visual_adapter=True, visual_dim=4096, visual_grid=8, visual_layers=[2, 5, 8])
    data.update(conditioning_input='visual_tokens', task='joint')
    batch = 64
    steps = math.ceil(audit['unique_targets']['train']*60/batch)
    data['train'].update(trainable='visual_adapter', batch_size=batch, accumulation=1,
        steps=steps, warmup=50, lr=1e-4, eval_every=math.ceil(audit['unique_targets']['train']*10/batch),
        val_samples=64, task_mix={'text': 0., 'reference': .2, 'joint': .8, 'unconditional': 0.})
    cfg = Experiment.from_dict(data)
    if (root/'config.yaml').exists():
        raise ValueError('Configuration already exists; resume the existing experiment')
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    atomic_json(root/'protocol.json', {'architecture': 'Qwen frozen 8x8 spatial tokens + independent attention in blocks 3/6/9',
        'base_step': base['step'], 'training': 'target-uniform then FLUX/Qwen variant; 80% joint / 20% image-only',
        'planned_exposures': 60, 'milestone_exposures': [10, 30, 60],
        'text_supervision': 'unchanged filename-only labels and exact existing text features',
        'augmentation': 'none for the first feature-path experiment; contain resize with white letterbox',
        'protection': 'hard bypass without reference, frozen backbone digest, exact generation regression',
        'model_selection': 'held-out validation only; image panels also required for quality assessment',
        'test_used': False, 'phase_b_lora': False})
    print('PREPARED', steps, 'updates before batch probe', flush=True)


class Data:
    def __init__(self, root, partial=False):
        selection_file = root/'selection.json'
        self.selection_sha256 = file_hash(selection_file) if selection_file.exists() else None
        if self.selection_sha256 is not None:
            # Retain original feature indices; a reviewed subset may have no
            # selected samples in the first cache chunk used by ordinary probes.
            partial = False
        self.audit = json.loads((root/'data_audit.json').read_text())
        self.cfg = Experiment.load(root/'config.yaml')
        pairs_hash = file_hash(root/'pairs.json')
        if pairs_hash != self.audit['pairs_sha256']:
            raise ValueError('Pairs changed')
        meta = json.loads((root/'visual_features/metadata.json').read_text())
        if meta['pairs_sha256'] != pairs_hash or (not partial and not meta['complete']):
            raise ValueError('Spatial cache incomplete or mismatched')
        self.visual_identity = meta['visual_identity']
        self.features = np.memmap(root/'visual_features/features.uint16.mmap', mode='r',
                                   dtype='uint16', shape=tuple(meta['shape']))
        if self.features.shape[1:] != (self.cfg.model.visual_grid**2, self.cfg.model.visual_dim):
            raise ValueError('Visual cache shape differs from model')
        if not partial and set(meta['chunks']) != {str(i) for i in range(0, meta['shape'][0], 64)}:
            raise ValueError('Missing spatial cache chunk hashes')
        chunks = {'0': meta['chunks']['0']} if partial else meta['chunks']
        for offset, digest in chunks.items():
            start = int(offset)
            if hashlib.sha256(np.asarray(self.features[start:start+64]).tobytes()).hexdigest() != digest:
                raise ValueError('Spatial cache bytes changed')
        self.rows = json.loads((root/'pairs.json').read_text())
        allowed = selected_indices(selection_file, root/'pairs.json', self.rows) if self.selection_sha256 else None
        if partial:
            self.rows = self.rows[:64]
        if not partial and len(self.rows) != meta['shape'][0]:
            raise ValueError('Missing visual rows')
        if any(r['feature_index'] != i for i, r in enumerate(self.rows)):
            raise ValueError('Visual feature row order changed')
        targets, maps = [], {}
        for row in self.rows:
            spec = row['target']
            if spec['path'] not in maps:
                maps[spec['path']] = np.memmap(spec['path'], mode='r', dtype='uint8', shape=tuple(spec['shape']))
            array = np.array(maps[spec['path']][spec['index']])
            if content_hash(array) != row['target_hash']:
                raise ValueError('MC target changed')
            targets.append(encode(array))
        self.targets = torch.stack(targets)
        self.encoder = DiskTextEncoder(TEXT_CACHE, expected_identity=self.audit['base_text_identity'])
        self.text, self.mask = self.encoder.encode(conditioned_prompts([r['prompt'] for r in self.rows], self.cfg), 'cpu')
        self.groups = {}
        for split in ('train', 'val'):
            groups = defaultdict(list)
            for i, row in enumerate(self.rows):
                if row['split'] == split and (allowed is None or i in allowed):
                    groups[row['target_hash']].append(i)
            self.groups[split] = list(groups.values())
        train_hashes = {self.rows[g[0]]['target_hash'] for g in self.groups['train']}
        if train_hashes.intersection(self.rows[g[0]]['target_hash'] for g in self.groups['val']):
            raise ValueError('Target split overlap')

    def draw(self, n, rng, split='train'):
        groups = self.groups[split]
        choices = torch.randint(len(groups), (n,), generator=rng).tolist()
        return [groups[i][int(torch.randint(len(groups[i]), (), generator=rng))] for i in choices]

    def batch(self, ids, device):
        refs = torch.from_numpy(np.array(self.features[ids], copy=True)).view(torch.bfloat16).to(device)
        return self.targets[ids].to(device), self.text[ids].to(device), self.mask[ids].to(device), refs


def objective(model, batch, device, seed, joint_fraction=.8):
    target, text, mask, refs = batch
    rng = torch.Generator(device=device).manual_seed(seed)
    keep_text = torch.rand(len(target), generator=rng, device=device) < joint_fraction
    mask = mask & keep_text[:, None]
    t = torch.rand(len(target), generator=rng, device=device)
    noise = torch.randn(target.shape, generator=rng, device=device)
    noisy = target*(1-t[:, None, None, None]) + noise*t[:, None, None, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(noisy, t, model.context(text, mask, refs, None))
        return F.mse_loss(pred.float(), noise-target)


def smoke(root, device):
    data = Data(root, partial=True)
    base = load_checkpoint(data.audit['base_checkpoint'])
    torch.manual_seed(data.cfg.train.seed)
    model = initialize_visual_adapter(data.cfg, base, device)
    original_digest = backbone_digest(model)
    ids = [g[0] for g in data.groups['train'][:16]]
    batch = data.batch(ids, device)
    text, mask = batch[1:3]
    seeds = list(range(len(ids)))
    with torch.no_grad():
        original = sample(model, text, mask, seeds=seeds, steps=4).cpu()
        initial = objective(model, batch, device, 7919).item()
    opt = torch.optim.AdamW(model.visual_adapter.parameters(), lr=3e-4)
    start = time.monotonic()
    for k in range(80):
        opt.zero_grad(set_to_none=True)
        loss = objective(model, batch, device, 10007+k)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.visual_adapter.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if k % 20 == 19:
            print('SMOKE', k+1, loss.item(), flush=True)
    with torch.no_grad():
        final = objective(model, batch, device, 7919).item()
        exact = torch.equal(original, sample(model, text, mask, seeds=seeds, steps=4).cpu())
    projection_grad = model.visual_adapter.projection[-1].weight.grad.norm().item()
    report = {'passed': final < initial and exact and backbone_digest(model) == original_digest and projection_grad > 0,
        'updates': 80, 'targets': len(ids), 'held_noise_loss_before': initial, 'held_noise_loss_after': final,
        'text_generation_exact': exact, 'backbone_exact': backbone_digest(model) == original_digest,
        'projection_gradient_norm': projection_grad,
        'gate_mean_abs': {k: g.detach().abs().mean().item() for k, g in model.visual_adapter.gates.items()},
        'seconds': time.monotonic()-start, 'weights_reused': False}
    atomic_json(root/'smoke.json', report)
    print('SMOKE RESULT', report, flush=True)
    if not report['passed']:
        raise RuntimeError('Spatial reference learning/preservation smoke failed')


def probe(root, device):
    data = Data(root, partial=True)
    base = load_checkpoint(data.audit['base_checkpoint'])
    model = initialize_visual_adapter(data.cfg, base, device)
    opt = torch.optim.AdamW(model.visual_adapter.parameters(), lr=1e-4)
    results = []
    free, total = torch.cuda.mem_get_info(device)
    for n in (32, 64, 96, 128, 160, 192):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for k in range(5):
                ids = data.draw(n, torch.Generator().manual_seed(k))
                batch = data.batch(ids, device)
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                start = time.monotonic()
                loss = objective(model, batch, device, 104729+k)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.visual_adapter.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize(device)
                times.append(time.monotonic()-start)
            duration = statistics.median(times[1:])
            row = {'batch_size': n, 'seconds_per_step': duration, 'images_per_second': n/duration,
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
        raise RuntimeError('No batch fits alongside existing GPU services')
    fastest = max(r['images_per_second'] for r in valid)
    # Shared GPU timings fluctuate: within 3% of best, prefer the smaller batch.
    chosen = min((r for r in valid if r['images_per_second'] >= fastest*.97), key=lambda r: r['batch_size'])
    cfg = data.cfg
    cfg.train.batch_size = chosen['batch_size']
    targets = len(data.groups['train'])
    cfg.train.steps = math.ceil(targets*60/cfg.train.batch_size)
    cfg.train.warmup = min(cfg.train.warmup, cfg.train.steps-1)
    cfg.train.eval_every = math.ceil(targets*10/cfg.train.batch_size)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    atomic_json(root/'batch_budget.json', {'measurements': results, 'chosen': chosen,
        'free_gib_before': free/1024**3, 'total_gib': total/1024**3, 'allocator_limit_fraction': .14,
        'steps': cfg.train.steps, 'conditional_draws_per_target': cfg.train.steps*cfg.train.batch_size/targets,
        'estimated_training_seconds': cfg.train.steps*chosen['seconds_per_step'],
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'note': 'Existing inference services remain running; timing includes shared GPU contention.'})


def choose_panels(data):
    selected = []
    for split in ('train', 'val'):
        used = set()
        for category in ('helmet', 'sword', 'pickaxe', 'shield', 'gem', 'axe'):
            candidates = [g[0] for g in data.groups[split]
                          if data.rows[g[0]]['prompt'].split()[-1] == category and g[0] not in used]
            if candidates:
                used.add(candidates[0])
                selected.append(candidates[0])
        # Reviewed sets may contain only custom names outside these categories.
        for group in data.groups[split]:
            if len(used) >= min(6, len(data.groups[split])):
                break
            if group[0] not in used:
                used.add(group[0])
                selected.append(group[0])
    return selected


def thumbnail(im, size=112, native=False):
    im = im.convert('RGBA')
    background = Image.new('RGBA', im.size, '#edf0f4')
    background.alpha_composite(im)
    result = ImageOps.contain(background.convert('RGB'), (size, size),
                            Image.Resampling.NEAREST if native else Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), '#edf0f4')
    canvas.paste(result, ((size-result.width)//2, (size-result.height)//2))
    return canvas


@torch.no_grad()
def generation_checks(model, data, root, step, device, panel_ids, original_text):
    out = root/'generation_checks'/f'{step:06d}'
    out.mkdir(parents=True, exist_ok=True)
    _, text, mask, refs = data.batch(panel_ids, device)
    seeds = [20261105]*len(panel_ids)
    text_result = sample(model, text, mask, seeds=seeds, steps=24).cpu()
    if not torch.equal(text_result, original_text):
        raise RuntimeError('Original text generation changed')
    results = {'text': text_result}
    for mode, active_mask in [('image', torch.zeros_like(mask)), ('joint', mask)]:
        results[mode] = sample(model, text, active_mask, refs, seeds=seeds, steps=24).cpu()
    results['joint_seed2'] = sample(model, text, mask, refs, seeds=[s+1 for s in seeds], steps=24).cpu()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    canvas = Image.new('RGB', (760, 55+len(panel_ids)*140), '#111827')
    draw = ImageDraw.Draw(canvas)
    labels = ['HD reference', 'MC target', 'Text base', 'Reference only', 'Text + reference', 'Second seed']
    for c, label in enumerate(labels):
        draw.text((12+c*124, 16), label, fill='white', font=font)
    for j, i in enumerate(panel_ids):
        row = data.rows[i]
        y = 43+j*140
        with Image.open(row['reference_file']) as im:
            pictures = [thumbnail(im), thumbnail(Image.fromarray(decode(data.targets[i])), native=True)]
        for mode in ('text', 'image', 'joint', 'joint_seed2'):
            image = Image.fromarray(decode(results[mode][j]))
            image.save(out/f'{j:02d}_{mode}.png')
            pictures.append(thumbnail(image, native=True))
        for c, picture in enumerate(pictures):
            canvas.paste(picture, (12+c*124, y))
        draw.text((12, y+115), f'{row["split"]}: {row["prompt"]}', fill='white', font=font)
    canvas.save(out/'comparison.png')
    # Fixed generic text and seed across different target/reference designs.
    diversity = []
    diversity_rows = []
    for category in ('helmet', 'sword', 'shield', 'axe'):
        ids = [g[0] for g in data.groups['val'] if data.rows[g[0]]['prompt'].split()[-1] == category][:4]
        if len(ids) < 2:
            continue
        h, m = data.encoder.encode(conditioned_prompts([category]*len(ids), data.cfg), device)
        r = data.batch(ids, device)[3]
        plain = sample(model, h, m, seeds=[20261105]*len(ids), steps=24).cpu()
        variations = []
        for seed in (20261105, 20261106):
            output = sample(model, h, m, r, seeds=[seed]*len(ids), steps=24).cpu()
            variations.append(output)
            for j, i in enumerate(ids):
                name = f'diversity_{category}_{j}_seed{seed}.png'
                Image.fromarray(decode(output[j])).save(out/name)
                diversity.append({'prompt': category, 'seed': seed, 'output': name,
                                  'target_hash': data.rows[i]['target_hash'],
                                  'reference_file': data.rows[i]['reference_file']})
        for j, i in enumerate(ids):
            diversity_rows.append((category, i, plain[j], variations[0][j], variations[1][j]))
    sheet = Image.new('RGB', (640, 55+140*len(diversity_rows)), '#111827')
    painter = ImageDraw.Draw(sheet)
    for c, label in enumerate(('HD reference', 'MC target', 'Fixed text base', 'Joint seed 1', 'Joint seed 2')):
        painter.text((12+c*126, 16), label, fill='white', font=font)
    for j, (category, i, plain, first, second) in enumerate(diversity_rows):
        y = 43+j*140
        with Image.open(data.rows[i]['reference_file']) as im:
            pictures = [thumbnail(im), thumbnail(Image.fromarray(decode(data.targets[i])), native=True)]
        pictures.extend(thumbnail(Image.fromarray(decode(v)), native=True) for v in (plain, first, second))
        for c, picture in enumerate(pictures):
            sheet.paste(picture, (12+c*126, y))
        painter.text((12, y+115), f'fixed text: {category} | {data.rows[i]["prompt"]}', fill='white', font=font)
    sheet.save(out/'same_text_different_references.png')
    atomic_json(out/'diversity.json', diversity)
    atomic_json(out/'panel.json', [{'row': j, 'pair_index': i, **data.rows[i]} for j, i in enumerate(panel_ids)])


def train(root, device, stop_after=None):
    data = Data(root)
    cfg = data.cfg
    base = load_checkpoint(data.audit['base_checkpoint'])
    if file_hash(data.audit['base_checkpoint']) != data.audit['base_checkpoint_sha256']:
        raise ValueError('Changed base checkpoint')
    torch.manual_seed(cfg.train.seed)
    model = initialize_visual_adapter(cfg, base, device)
    digest = backbone_digest(model)
    opt = torch.optim.AdamW(model.visual_adapter.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    out = root/'training'
    out.mkdir(exist_ok=True)
    step, best, best_image = 0, float('inf'), float('inf')
    if (out/'latest.pt').exists():
        state = load_checkpoint(out/'latest.pt')
        if (Experiment.from_dict(state['config']).to_dict() != cfg.to_dict()
                or state['visual_identity'] != data.visual_identity
                or state.get('selection_sha256') != data.selection_sha256
                or state['catalog_fingerprint'] != data.audit['pairs_sha256']):
            raise ValueError('Resume contract changed')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        step, best, best_image = state['step'], state['best_joint'], state['best_image']
    if backbone_digest(model) != digest:
        raise RuntimeError('Backbone mismatch after initialization/resume')
    panel_ids = choose_panels(data)
    _, panel_text, panel_mask, _ = data.batch(panel_ids, device)
    # CUDA can select slightly different kernels for trainable vs frozen
    # parameters even under no_grad. Compare two equally frozen generators.
    original = PixelFlow(Experiment.from_dict(base['config']).model).to(device).eval().requires_grad_(False)
    original.load_state_dict(base['model'], strict=True)
    with torch.no_grad():
        original_text = sample(original, panel_text, panel_mask, seeds=[20261105]*len(panel_ids), steps=24).cpu()
    del original, base
    torch.cuda.empty_cache()
    val_ids = [g[0] for g in data.groups['val']]
    rng = torch.Generator().manual_seed(20261105)
    val_ids = [val_ids[i] for i in torch.randperm(len(val_ids), generator=rng)[:cfg.train.val_samples].tolist()]
    val_batch = data.batch(val_ids, device)
    target, text, mask, refs = val_batch
    atomic_json(root/'run.json', {'base_checkpoint': data.audit['base_checkpoint'], 'base_sha256': data.audit['base_checkpoint_sha256'],
        'backbone_sha256': digest, 'visual_identity': data.visual_identity, 'text_identity': data.audit['base_text_identity'],
        'selection_sha256': data.selection_sha256,
        'training_targets': len(data.groups['train']), 'validation_targets': len(data.groups['val']),
        'validation_pair_indices': val_ids, 'panel_pair_indices': panel_ids,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable': 'visual projection, reference attention and gates only', 'config': cfg.to_dict()})
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    start = time.monotonic()
    start_step = step

    def save(name):
        save_checkpoint(out/name, {'config': cfg.to_dict(), 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'step': step, 'best_joint': best, 'best_image': best_image,
            'text_identity': data.audit['base_text_identity'], 'visual_identity': data.visual_identity,
            'catalog_fingerprint': data.audit['pairs_sha256'], 'backbone_sha256': digest,
            'selection_sha256': data.selection_sha256,
            'initialization': {'path': data.audit['base_checkpoint'], 'sha256': data.audit['base_checkpoint_sha256'], 'step': 4128}})

    @torch.no_grad()
    def validate():
        nonlocal best, best_image
        if backbone_digest(model) != digest:
            raise RuntimeError('Original generator tensors changed')
        scores = {'step': step, 'exposures': step*cfg.train.batch_size/len(data.groups['train']), 'backbone_exact': True}
        for mode, active_mask in [('joint', mask), ('image', torch.zeros_like(mask))]:
            losses = []
            for k, level in enumerate((.1, .5, .9)):
                gen = torch.Generator(device=device).manual_seed(104729+k)
                noise = torch.randn(target.shape, generator=gen, device=device)
                noisy = target*(1-level)+noise*level
                t = torch.full((len(target),), level, device=device)
                parts = {}
                for label, reference in [('matched', refs), ('wrong_reference_same_text', refs.roll(1, 0)), ('no_reference', None)]:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(noisy, t, model.context(text, active_mask, reference, None))
                    parts[label] = F.mse_loss(pred.float(), noise-target).item()
                parts['reference_gain'] = parts['wrong_reference_same_text']-parts['matched']
                scores[f'{mode}_t{level}'] = parts
                losses.append(parts['matched'])
            scores[f'{mode}_flow_mse'] = sum(losses)/len(losses)
        scores['gate_mean_abs'] = {k: g.detach().abs().mean().item() for k, g in model.visual_adapter.gates.items()}
        generation_checks(model, data, root, step, device, panel_ids, original_text)
        scores['text_generation_exact'] = True
        if scores['joint_flow_mse'] < best:
            best = scores['joint_flow_mse']
            save('best_joint_flow.pt')
        if scores['image_flow_mse'] < best_image:
            best_image = scores['image_flow_mse']
            save('best_image_flow.pt')
        save('latest.pt')
        with (out/'validation.jsonl').open('a') as f:
            f.write(json.dumps(scores)+'\n')
        atomic_json(root/'validation_latest.json', scores)
        print('VALIDATION', json.dumps(scores), flush=True)

    if step == 0:
        save('initial_adapter.pt')
        validate()
    end = min(cfg.train.steps, stop_after) if stop_after is not None else cfg.train.steps
    milestones = {math.ceil(len(data.groups['train'])*exposure/cfg.train.batch_size): exposure for exposure in (10, 30, 60)}
    print('TRAINING', step, end, 'batch', cfg.train.batch_size, flush=True)
    for update in range(step, end):
        if stopping[0]:
            break
        rng = torch.Generator().manual_seed(cfg.train.seed+1000003*update)
        ids = data.draw(cfg.train.batch_size, rng)
        batch = data.batch(ids, device)
        warm = min(1., (update+1)/max(1, cfg.train.warmup))
        progress = max(0., (update-cfg.train.warmup)/max(1, cfg.train.steps-cfg.train.warmup))
        lr = cfg.train.lr*warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss = objective(model, batch, device, cfg.train.seed+update+17)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite spatial adapter loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.visual_adapter.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
        opt.step()
        step = update+1
        elapsed = time.monotonic()-start
        metric = {'step': step, 'loss': loss.item(), 'lr': lr, 'gradient_norm': norm.item(),
            'seconds': elapsed, 'exposures': step*cfg.train.batch_size/len(data.groups['train'])}
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric)+'\n')
        atomic_json(root/'status.json', {**metric, 'planned_steps': cfg.train.steps, 'complete': False,
            'estimated_remaining_seconds': elapsed/max(1, step-start_step)*(cfg.train.steps-step)})
        if step % 20 == 0:
            print('STEP', step, 'loss', round(loss.item(), 6), 'seconds', round(elapsed, 1), flush=True)
        if step % cfg.train.eval_every == 0 or step in milestones or step == end:
            validate()
            if step in milestones:
                save(f'exposure{milestones[step]:02d}.pt')
    save('latest.pt')
    atomic_json(root/'status.json', {'step': step, 'planned_steps': cfg.train.steps, 'complete': step == cfg.train.steps,
        'stopped': stopping[0], 'seconds': time.monotonic()-start, 'best_joint': best, 'best_image': best_image,
        'exposures': step*cfg.train.batch_size/len(data.groups['train']), 'backbone_exact': backbone_digest(model) == digest})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'smoke', 'probe', 'train'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.stage == 'prepare':
        prepare(a.out)
    else:
        torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(.14, a.device)
        if a.stage == 'smoke':
            smoke(a.out, a.device)
        elif a.stage == 'probe':
            probe(a.out, a.device)
        else:
            train(a.out, a.device, a.stop_after)
