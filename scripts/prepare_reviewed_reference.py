"""Prepare a separate spatial-reference run that trains only on reviewed selections."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.pair_selection import selected_indices


def prepare(selection, source, out, steps=None):
    source, out, selection = source.resolve(), out.resolve(), selection.resolve()
    if out.exists():
        raise ValueError('Use a new output directory; existing experiments are never overwritten')
    rows = json.loads((source/'pairs.json').read_text())
    chosen = selected_indices(selection, source/'pairs.json', rows)
    cfg = Experiment.load(source/'config.yaml')
    if not cfg.model.visual_adapter:
        raise ValueError('Expected an existing spatial-reference experiment')
    meta = json.loads((source/'visual_features/metadata.json').read_text())
    if not meta['complete'] or meta['pairs_sha256'] != file_hash(source/'pairs.json'):
        raise ValueError('Source spatial cache is incomplete or mismatched')
    targets = {s: len({rows[i]['target_hash'] for i in chosen if rows[i]['split'] == s}) for s in ('train', 'val')}
    counts = {s: sum(rows[i]['split'] == s for i in chosen) for s in ('train', 'val')}
    if steps is not None:
        cfg.train.steps = steps
    cfg.train.warmup = min(cfg.train.warmup, cfg.train.steps-1)
    cfg.train.eval_every = min(cfg.train.eval_every, cfg.train.steps)
    cfg.train.val_samples = min(cfg.train.val_samples, targets['val'])
    Experiment.from_dict(cfg.to_dict())
    out.mkdir(parents=True)
    shutil.copy2(source/'pairs.json', out/'pairs.json')
    shutil.copy2(selection, out/'selection.json')
    audit = json.loads((source/'data_audit.json').read_text())
    audit['reviewed_selection'] = {'sha256': file_hash(out/'selection.json'), 'counts': counts, 'unique_targets': targets}
    atomic_json(out/'data_audit.json', audit)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    (out/'visual_features').symlink_to((source/'visual_features').resolve(), target_is_directory=True)
    budget_file = source/'batch_budget.json'
    if budget_file.exists():
        budget = json.loads(budget_file.read_text())
        budget.update(steps=cfg.train.steps, conditional_draws_per_target=cfg.train.steps*cfg.train.batch_size/targets['train'],
                      selection_sha256=file_hash(out/'selection.json'), original_fixed_shape_batch_probe_reused=True)
        budget['estimated_training_seconds'] = cfg.train.steps*budget['chosen']['seconds_per_step']
        atomic_json(out/'batch_budget.json', budget)
    atomic_json(out/'reviewed_run.json', {'source': str(source), 'selection_sha256': file_hash(out/'selection.json'),
        'counts': counts, 'unique_targets': targets, 'cache_indices_unchanged': True,
        'source_experiment_modified': False, 'training_started': False,
        'steps': cfg.train.steps, 'batch_size': cfg.train.batch_size})
    print(json.dumps({'output': str(out), 'counts': counts, 'targets': targets,
        'next_command': f'PYTHONPATH=src python scripts/train_visual_reference.py train --out {out} --device cuda:0'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/spatial_reference_20261104'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    prepare(args.selection, args.source, args.out, args.steps)
