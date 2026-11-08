"""Prepare balanced five-view continuation without changing images or labels."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil

import yaml

from mcgen.catalog import atomic_json, file_hash, read_catalog, write_catalog
from mcgen.config import Experiment
from mcgen.five_task_reference import FiveTaskData, TASK_VIEWS
from mcgen.runtime import load_checkpoint

SOURCE = Path('outputs/reference_recovery_20261108')
DEFAULT = Path('outputs/reference_five_task_20261108')


def prepare(out, checkpoint=None, freeze_time_mlp=False, *, text_replay=False,
            replay_catalog=Path('outputs/qwen_vl_embedding_large_20261103/catalog'),
            replay_features=Path('outputs/qwen_vl_embedding_large_20261103/features'),
            replay_text=144, replay_null=16, replay_targets=2000, steps=None):
    source = SOURCE.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError('Choose a fresh experiment directory')
    if text_replay and (not freeze_time_mlp or replay_text < 1 or replay_null < 0):
        raise ValueError('Broad replay requires a frozen time MLP and positive text budget')
    steps = steps if steps is not None else (8000 if text_replay else 16000)
    if steps < 500:
        raise ValueError('Training budget must include the first 500-step validation')
    status = json.loads((source/'status.json').read_text())
    if not status['complete'] or status['step'] != 8000:
        raise ValueError('The full8000 source must be complete')
    out.mkdir(parents=True)
    for name in ('pairs.json', 'review_snapshot.json', 'data_audit.json', 'excluded_pairs.json'):
        shutil.copy2(source/name, out/name)
    for name in ('text_features', 'visual_features'):
        (out/name).symlink_to((source/name).resolve(), target_is_directory=True)
    cfg = Experiment.load(source/'config.yaml')
    cfg.train.steps = steps
    cfg.train.batch_size = 160+(replay_text+replay_null if text_replay else 0)
    cfg.train.accumulation = 1  # Actual microbatching is benchmarked, logical batch is fixed.
    cfg.train.lr = 2e-6
    cfg.train.warmup = 300
    cfg.train.eval_every = 2000
    nt, nu = (replay_text, replay_null) if text_replay else (0, 0)
    cfg.train.task_mix = {'joint': 62/cfg.train.batch_size, 'text': (62+nt)/cfg.train.batch_size,
        'reference': 31/cfg.train.batch_size, 'unconditional': (5+nu)/cfg.train.batch_size}
    (out/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else source/'training/latest.pt'
    state = load_checkpoint(checkpoint)
    if state.get('diagnostic_only') or state.get('target_leakage') or state['step'] != 8000:
        raise ValueError('Invalid production initializer')
    identities = {'text': state['text_identity'], 'visual': state['visual_identity']}
    del state
    gc.collect()
    replay_report = None
    if text_replay:
        from mcgen.disk_text import DiskTextEncoder
        from mcgen.text_replay import select_replay_records, stratified_text_subset
        from mcgen.tasks import conditioned_prompts
        records, parent = read_catalog(replay_catalog)
        if parent['native_size'] != cfg.model.size:
            raise ValueError('Replay must use the original native16 catalog')
        pairs = json.loads((out/'pairs.json').read_text())
        selected, replay_report = select_replay_records(records, pairs)
        del records
        selected, subset_report = stratified_text_subset(selected, replay_targets)
        replay_report['eligible_split_targets'] = replay_report['split_targets']
        replay_report.update(subset=subset_report,
            split_targets={s: len({r['target_hash'] for r in selected if r['split'] == s}) for s in ('train', 'val')},
            train_categories=subset_report['train_categories'], train_unique_prompts=subset_report['train_unique_prompts'])
        encoder = DiskTextEncoder(replay_features, identities['text'])
        if any(p not in encoder.index for p in conditioned_prompts([r['prompt'] for r in selected], cfg)):
            raise ValueError('Replay catalog prompts are missing from the original embedding cache')
        meta = write_catalog(out/'text_replay_catalog', selected, {'native_size': cfg.model.size,
            'parent_catalog': str(replay_catalog.resolve()), 'parent_fingerprint': parent['fingerprint'],
            'supervision': 'unchanged original deterministic filename text',
            'selection': 'original splits; exclude paired target hashes and reference holdout projects',
            'reference_pairs_sha256': file_hash(out/'pairs.json')})
        atomic_json(out/'text_replay.json', {'catalog': str(out/'text_replay_catalog'),
            'catalog_fingerprint': meta['fingerprint'], 'source_catalog_fingerprint': parent['fingerprint'],
            'features': str(replay_features.resolve()), 'text_identity': encoder.identity,
            'feature_hashes': {n: file_hash(replay_features/n) for n in
                ('metadata.json', 'prompts.json', 'features.uint16.mmap', 'masks.uint8.mmap')},
            'sampling': 'uniform target, then uniform existing caption; no generated labels',
            'audit': replay_report})
        atomic_json(out/'text_replay_selection.json', [r for r in selected if r['split'] == 'train'])
        del selected, encoder
        gc.collect()
    atomic_json(out/'initialization.json', {'checkpoint': str(checkpoint), 'sha256': file_hash(checkpoint),
        'source_run': str(source), 'source_checkpoint_run': str(checkpoint.parent.parent),
        'source_step': 8000, 'transfer': 'complete source model; optimizer reset',
        'encoders_frozen': True, 'all_generator_and_visual_parameters_trainable_from_start': not freeze_time_mlp,
        'freeze_time_mlp': freeze_time_mlp,
        'diagnostic_weights_used': False})
    audit = json.loads((out/'data_audit.json').read_text())
    recipe = {'version': 'five_task_broad_replay_v1' if text_replay else 'five_task_reference_v2',
        'steps': steps, 'batch_size': cfg.train.batch_size,
        'pairs_per_update': 31, 'unconditional_per_update': 5,
        'views': [{'name': n, 'task': t, 'variant': v, 'per_update': 31} for n, t, v in TASK_VIEWS],
        'source_hq_fraction': .6, 'target_uniform_within_source': True,
        'base_lr': 2e-6, 'visual_lr': 2e-5, 'warmup_steps': 300,
        'decay_start': int(.75*steps), 'lr_floor': .3, 'gradient_clip': 1.,
        'evaluate_updates': sorted({n for n in (0, 500, 1000, 2000, 4000, 8000, 12000, steps) if n <= steps}),
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
    if text_replay:
        recipe.update(replay_text_per_update=nt, replay_null_per_update=nu,
            text_replay_sha256=file_hash(out/'text_replay.json'),
            broad_text_generation_samples=24,
            broad_text_flow_samples=1024,
            gradient_accumulation='per-example sum divided by logical batch; all tasks in each optimizer update',
            budget_note='8000 additional updates by default; monitor fixed validation from update0 and retain separate task bests',
            checkpoint_selection='text best uses broad validation generation plus paired text; reference/balanced best also retained',
            joint_text_control_limitation='Replay preserves broad T2I; it does not create text-essential joint edit supervision.')
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
        'updates': steps, 'logical_batch': cfg.train.batch_size, 'total_draws': steps*cfg.train.batch_size,
        'draws_per_paired_task': steps*31, 'text_only_draws': steps*(62+nt), 'source_step': 8000,
        'task_mix': cfg.train.task_mix, 'text_replay': replay_report}
    if text_replay:
        info.update(broad_text_draws=steps*nt, broad_null_draws=steps*nu,
            broad_text_exposure_per_target=steps*nt/replay_report['split_targets']['train'])
    atomic_json(out/'preparation.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--checkpoint', type=Path, help='Healthy complete8000-step model, e.g. five-task best_text.pt')
    parser.add_argument('--freeze-time-mlp', action='store_true')
    parser.add_argument('--text-replay', action='store_true', help='Add the original broad filename-only T2I train set')
    parser.add_argument('--replay-catalog', type=Path, default=Path('outputs/qwen_vl_embedding_large_20261103/catalog'))
    parser.add_argument('--replay-features', type=Path, default=Path('outputs/qwen_vl_embedding_large_20261103/features'))
    parser.add_argument('--replay-text', type=int, default=144)
    parser.add_argument('--replay-null', type=int, default=16)
    parser.add_argument('--replay-targets', type=int, default=2000, help='Fixed diverse unique training targets from the original catalog')
    parser.add_argument('--steps', type=int, help='New optimizer updates; default8000 with replay, otherwise16000')
    args = parser.parse_args()
    prepare(args.out, args.checkpoint, args.freeze_time_mlp, text_replay=args.text_replay,
        replay_catalog=args.replay_catalog, replay_features=args.replay_features,
        replay_text=args.replay_text, replay_null=args.replay_null, replay_targets=args.replay_targets, steps=args.steps)
