from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from mcgen.five_task_reference import FiveTaskSampler, TASK_VIEWS, TaskPlan
from mcgen.text_replay import (append_replay, merge_task_batches, select_replay_records,
                              stratified_text_subset, TextReplayData)


def record(key, split='train', project=None):
    return {'id': key, 'split': split, 'target_hash': key, 'split_group': project or key,
            'lineage': project or key, 'prompt': key+' helmet', 'asset_type': 'item',
            'caption_supervision': 'deterministic_resource_text_only'}


def test_replay_excludes_paired_targets_and_entire_holdout_project_without_relabeling():
    original = [record('train'), record('holdout', project='held_project'),
                record('held_sibling', project='held_project'), record('paired'),
                record('val', 'val'), record('test', 'test')]
    pairs = [{'target_hash': 'holdout', 'split': 'val', 'split_group': 'different_source_group'},
             {'target_hash': 'paired', 'split': 'train', 'split_group': 'paired_source'}]
    kept, audit = select_replay_records(original, pairs)
    assert [r['id'] for r in kept] == ['train', 'val']
    assert kept[0]['prompt'] == original[0]['prompt'] and kept[1]['split'] == 'val'
    assert audit['reference_holdout_targets_seen_by_foundation'] == 1
    assert audit['excluded_records']['reference_holdout_project'] == 1
    assert audit['excluded_records']['paired_target_overlap'] == 2


def test_bad_supervision_is_not_silently_used_as_filename_text():
    rows = [record('train'), record('val', 'val')]
    rows[0]['caption_supervision'] = 'visual_caption'
    with pytest.raises(ValueError, match='filename-only'):
        select_replay_records(rows, [])


def test_broad_replay_keeps_five_views_and_exact_counts_and_is_reproducible():
    rows = [{'target_hash': str(i), 'split': s, 'quality_tier': 'highest_quality'}
            for i, s in enumerate(('train', 'val', 'test'))]
    sampler = FiveTaskSampler(rows)
    replay = SimpleNamespace(sample=lambda n, rng: torch.randint(1, 100, (n,), generator=rng).tolist())
    def draw():
        rng = torch.Generator().manual_seed(239)
        return append_replay(sampler.sample(31, 5, rng), replay, 144, 16, rng)
    plan = draw()
    assert plan == draw() and len(plan) == 320
    assert Counter(plan.names) == {**{n: 31 for n, _, _ in TASK_VIEWS},
        'unconditional': 5, 'broad_text': 144, 'broad_unconditional': 16}
    assert Counter(plan.tasks) == {'joint': 62, 'text': 206, 'reference': 31, 'unconditional': 21}
    assert all(i == 0 for i, source in zip(plan.ids, plan.sources) if source == 'paired')
    for slot in range(31):
        assert {plan.names[j] for j, x in enumerate(plan.pair_slots) if x == slot} == {n for n, _, _ in TASK_VIEWS}


def test_shuffled_replay_rows_keep_target_caption_and_absent_reference_aligned():
    plan = TaskPlan([4, 9, 5], ['text', 'joint', 'unconditional'], [0]*3,
                    ['broad_text', 'joint_short', 'broad_unconditional'], [-1, 0, -1],
                    ['replay', 'paired', 'replay'])
    paired = (torch.full((1, 4, 2, 2), 9.), torch.full((1, 1, 8), 90.),
              torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 4, 8), torch.tensor([True]))
    replay = (torch.stack([torch.full((4, 2, 2), 4.), torch.full((4, 2, 2), 5.)]),
              torch.stack([torch.full((1, 8), 40.), torch.full((1, 8), 50.)]),
              torch.tensor([[True], [False]]), None, torch.tensor([False, False]))
    target, text, mask, ref, present = merge_task_batches(plan, paired, replay)
    assert target[:, 0, 0, 0].tolist() == [4, 9, 5]
    assert text[:, 0, 0].tolist() == [40, 90, 50]
    assert mask[:, 0].tolist() == [True, True, False]
    assert present.tolist() == [False, True, False]
    assert torch.count_nonzero(ref[[0, 2]]) == 0 and torch.all(ref[1] == 1)


def test_replay_sampling_is_target_uniform_even_with_many_caption_variants():
    data = TextReplayData.__new__(TextReplayData)
    data.groups = [[0], list(range(1, 101))]
    ids = data.sample(10000, torch.Generator().manual_seed(31))
    assert .48 < ids.count(0)/len(ids) < .52


def test_broad_validation_is_fixed_unique_and_category_diverse():
    data = TextReplayData.__new__(TextReplayData)
    rows = [{**record(str(i), 'val'), 'category_15': 'gem' if i < 2 else 'sword'} for i in range(10)]
    rows.append(dict(rows[0], id='duplicate-caption'))
    data.datasets = {'val': SimpleNamespace(records=rows)}
    ids = data.validation_indices(20)
    assert len(ids) == len(set(ids)) == 10
    assert [rows[i]['category_15'] for i in ids[:4]] == ['gem', 'sword', 'gem', 'sword']
    assert ids == data.validation_indices(20)


def test_fixed_subset_covers_types_and_distinct_names_without_touching_validation():
    rows = [{**record(str(i)), 'category_15': 'sword' if i < 100 else 'gem',
             'prompt': ('iron sword' if i < 95 else f'color{i} sword') if i < 100 else f'color{i} gem'}
            for i in range(200)]
    rows += [record('heldout', 'val')]
    selected, report = stratified_text_subset(rows, count=20)
    train = [r for r in selected if r['split'] == 'train']
    assert len({r['target_hash'] for r in train}) == 20
    assert Counter(r['category_15'] for r in train) == {'sword': 10, 'gem': 10}
    assert len({r['prompt'] for r in train if r['category_15'] == 'sword'}) == 6
    originals = {r['id']: r for r in rows}
    assert all(r['prompt'] == originals[r['id']]['prompt'] for r in train)
    assert selected[-1] == rows[-1] and report['text_changed'] is False
    assert stratified_text_subset(list(reversed(rows)), 20)[0][:-1] == selected[:-1]


def test_other_category_is_split_to_cover_food_and_components():
    rows = [{**record(str(i)), 'category_15': 'other',
             'prompt': f'material{i} '+('cookie' if i < 30 else 'circuit')}
            for i in range(60)]
    selected, report = stratified_text_subset(rows, 10)
    assert report['train_types'] == {'other:component': 5, 'other:food': 5}
    assert len(selected) == 10
