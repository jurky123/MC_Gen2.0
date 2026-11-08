"""Prepare balanced five-view continuation without changing images or labels."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil

import yaml

from mcgen.catalog import atomic_json, file_hash
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData, TASK_VIEWS
from mcgen.runtime import load_checkpoint

SOURCE = Path('outputs/reference_recovery_20261108')
DEFAULT = Path('outputs/reference_five_task_20261108')


def prepare(out, checkpoint=None, freeze_time_mlp=False):
    source = SOURCE.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError('Choose a fresh experiment directory')
    status = json.loads((source/'status.json').read_text())
    if not status['complete'] or status['step'] != 8000:
        raise ValueError('The full8000 source must be complete')
    out.mkdir(parents=True)
    for name in ('pairs.json', 'review_snapshot.json', 'data_audit.json', 'excluded_pairs.json'):
        shutil.copy2(source/name, out/name)
    for name in ('text_features', 'visual_features'):
        (out/name).symlink_to((source/name).resolve(), target_is_directory=True)
    cfg = Experiment.load(source/'config.yaml')
    cfg.train.steps = 16000
    cfg.train.batch_size = 160
    cfg.train.accumulation = 1  # Actual microbatching is benchmarked, logical batch is fixed.
    cfg.train.lr = 2e-6
    cfg.train.warmup = 300
    cfg.train.eval_every = 2000
    cfg.train.task_mix = {'joint': 62/160, 'text': 62/160, 'reference': 31/160, 'unconditional': 5/160}
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else source/'training/latest.pt'
    state = load_checkpoint(checkpoint)
    if state.get('diagnostic_only') or state.get('target_leakage') or state['step'] != 8000:
        raise ValueError('Invalid production initializer')
    identities = {'text': state['text_identity'], 'visual': state['visual_identity']}
    del state
    gc.collect()
    atomic_json(out/'initialization.json', {'checkpoint': str(checkpoint), 'sha256': file_hash(checkpoint),
        'source_run': str(source), 'source_checkpoint_run': str(checkpoint.parent.parent),
        'source_step': 8000, 'transfer': 'complete source model; optimizer reset',
        'encoders_frozen': True, 'all_generator_and_visual_parameters_trainable_from_start': not freeze_time_mlp,
        'freeze_time_mlp': freeze_time_mlp,
        'diagnostic_weights_used': False})
    audit = json.loads((out/'data_audit.json').read_text())
    recipe = {'version': 'five_task_reference_v2', 'steps': 16000, 'batch_size': 160,
        'pairs_per_update': 31, 'unconditional_per_update': 5,
        'views': [{'name': n, 'task': t, 'variant': v, 'per_update': 31} for n, t, v in TASK_VIEWS],
        'source_hq_fraction': .6, 'target_uniform_within_source': True,
        'base_lr': 2e-6, 'visual_lr': 2e-5, 'warmup_steps': 300,
        'decay_start': 12000, 'lr_floor': .3, 'gradient_clip': 1.,
        'evaluate_updates': [0, 500, 1000, 2000, 4000, 8000, 12000, 16000],
        'all_tasks_from_first_update': True, 'coarse_means': 'original filename/name caption, not a reduced noun',
        'freeze_time_mlp': freeze_time_mlp,
        'detail_means': 'existing verified MC description; explicit short-caption alias if absent',
        'joint_tasks_use_detail': True, 'same_pair_expanded_each_update': True,
        'independent_training_noise_per_view': True, 'gradient_accumulation': 'per-example sum divided by logical160',
        'monitor_precision': 'BF16 for five-mode panels; FP32 common text regression',
        'checkpoint_selection': 'HQ validation generation MAE balanced over all five views; text/reference best saved separately',
        'stability_guard': {'warmup': 300, 'calibration': 400, 'window': 100, 'gradient_clip': 1.,
            'gradient_ratio': 5., 'loss_ratio': 1.15, 'clipped_fraction': .8, 'patience': 3},
        'test_used_for_training_selection': False, 'source_pairs_sha256': audit['pairs_sha256'],
        'new_data_or_annotations': False, 'automatic_final_heldout_evaluation': True}
    atomic_json(out/'recipe.json', recipe)
    data = FiveTaskData(out)
    assert data.encoder.identity == identities['text'] and data.visual_identity == identities['visual']
    trained = [r for r in data.rows if r['split'] == 'train']
    captions = [{'id': r['id'], 'split': r['split'], 'short': r['captions'][0],
                 'detail': r['captions'][-1], 'has_distinct_detail': len(r['captions']) > 1,
                 'detail_policy': 'verified_existing' if len(r['captions']) > 1 else 'short_alias_not_fine_annotation'}
                for r in data.rows]
    atomic_json(out/'caption_policy.json', captions)
    info = {'complete': True, 'training_pairs': len(trained),
        'training_unique_targets': len({r['target_hash'] for r in trained}),
        'training_pairs_with_distinct_detail': sum(len(r['captions']) > 1 for r in trained),
        'training_pairs_with_short_alias': sum(len(r['captions']) == 1 for r in trained),
        'split_counts': dict(Counter(r['split'] for r in data.rows)),
        'same_data_labels_and_splits': file_hash(out/'pairs.json') == file_hash(source/'pairs.json'),
        'native_size': 16, 'lossy_target_count': audit['lossy_target_count'],
        'updates': 16000, 'logical_batch': 160, 'total_draws': 2560000,
        'draws_per_task': 496000, 'text_only_draws': 992000, 'source_step': 8000}
    atomic_json(out/'preparation.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--checkpoint', type=Path, help='Healthy complete8000-step model, e.g. five-task best_text.pt')
    parser.add_argument('--freeze-time-mlp', action='store_true')
    args = parser.parse_args()
    prepare(args.out, args.checkpoint, args.freeze_time_mlp)
