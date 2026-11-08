"""Human training selection persists, never leaks validation, and retains cache indices."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from mcgen.catalog import file_hash
from mcgen.pair_review import ReviewStore, ReviewConflict, create_review_app, target_hash
from mcgen.pair_selection import pair_id, selected_indices


@pytest.fixture
def dataset(tmp_path):
    pixels = np.zeros((2, 16, 16, 4), dtype=np.uint8)
    pixels[0, 3:12, 4:11] = [200, 40, 40, 255]
    pixels[1, 2:13, 5:10] = [40, 150, 80, 255]
    target = tmp_path/'targets.mmap'
    pixels.tofile(target)
    rows = []
    for n, split in enumerate(('train', 'val')):
        for generator in ('flux', 'qwen'):
            ref = tmp_path/f'{split}_{generator}.png'
            Image.new('RGB', (96, 64), (120+n*50, 80, 100 if generator == 'flux' else 180)).save(ref)
            rows.append({'id': f'{split}_{generator}', 'feature_index': len(rows), 'split': split,
                'target_hash': target_hash(pixels[n]), 'prompt': 'sun gem' if n == 0 else 'tin helmet',
                'source_filename': 'sun_gem.png' if n == 0 else 'tin_helmet.png',
                'reference_generator': generator, 'reference_file': str(ref), 'reference_sha256': file_hash(ref),
                'target': {'path': str(target), 'shape': list(pixels.shape), 'index': n}})
    pairs = tmp_path/'pairs.json'
    pairs.write_text(json.dumps(rows))
    return pairs, rows, tmp_path/'review'


def test_quality_is_independent_from_explicit_training_selection(dataset):
    pairs, rows, directory = dataset
    store = ReviewStore(pairs, directory)
    ident = pair_id(rows[0])
    store.update([ident], state='keep')
    assert store.summary()['counts']['train']['keep'] == 1
    assert store.summary()['train_selected_pairs'] == 0
    with pytest.raises(ValueError, match='还没有选入训练'):
        store.export()
    store.update([ident], train_selected=True)
    assert store.summary()['train_selected_pairs'] == 1
    store.update([ident], train_selected=False)
    assert store.summary()['counts']['train']['keep'] == 1
    assert store.summary()['train_selected_pairs'] == 0


def test_reject_deselects_and_undo_restores_both_fields(dataset):
    pairs, rows, directory = dataset
    store = ReviewStore(pairs, directory)
    ident = pair_id(rows[0])
    store.update([ident], state='keep', train_selected=True, note='keep the gem')
    store.update([ident], state='reject', reason='配对不一致')
    assert store.summary()['train_selected_pairs'] == 0
    store.undo()
    decision = store.snapshot()[0][ident]
    assert decision['train_selected'] and decision['state'] == 'keep' and decision['note'] == 'keep the gem'
    store.undo()
    assert ident not in store.snapshot()[0]


def test_validation_cannot_be_promoted_and_failed_batch_is_atomic(dataset):
    pairs, rows, directory = dataset
    store = ReviewStore(pairs, directory)
    with pytest.raises(ValueError, match='验证配对'):
        store.update([pair_id(rows[0]), pair_id(rows[2])], state='keep', train_selected=True)
    assert store.summary()['train_selected_pairs'] == 0
    assert store.snapshot()[1] == 0


def test_progress_survives_restart_and_conflicts_do_not_overwrite(dataset):
    pairs, rows, directory = dataset
    store = ReviewStore(pairs, directory)
    ident = pair_id(rows[0])
    store.update([ident], state='keep', train_selected=True, expected_revision=0)
    reopened = ReviewStore(pairs, directory)
    assert reopened.summary()['train_selected_pairs'] == 1
    with pytest.raises(ReviewConflict):
        reopened.update([ident], state='reject', expected_revision=0)
    assert reopened.summary()['train_selected_pairs'] == 1
    changed = pairs.with_name('changed.json')
    changed.write_text(pairs.read_text().replace('sun gem', 'different gem'))
    with pytest.raises(ValueError, match='different pair manifest'):
        ReviewStore(changed, directory)


def test_export_is_a_training_input_and_keeps_original_indices_and_splits(dataset):
    pairs, rows, directory = dataset
    before = file_hash(pairs)
    store = ReviewStore(pairs, directory)
    store.update([pair_id(rows[0])], state='keep', train_selected=True)
    store.update([pair_id(rows[1])], state='keep')
    store.update([pair_id(rows[3])], state='reject')
    result = store.export()
    assert result['train_pairs'] == 1 and result['val_pairs'] == 1
    selection = Path(result['selection_path'])
    assert selected_indices(selection, pairs, rows) == {0, 2}
    exported = [json.loads(s) for s in selection.with_name('selected_pairs.jsonl').read_text().splitlines()]
    assert [r['feature_index'] for r in exported] == [0, 2]
    assert [r['split'] for r in exported] == ['train', 'val']
    with zipfile.ZipFile(selection.with_name('selection.zip')) as archive:
        assert set(archive.namelist()) == {'selection.json', 'selected_pairs.jsonl', 'summary.json'}
    assert file_hash(pairs) == before


def test_changed_reference_prevents_export(dataset):
    pairs, rows, directory = dataset
    store = ReviewStore(pairs, directory)
    store.update([pair_id(rows[0])], state='keep', train_selected=True)
    Path(rows[0]['reference_file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='参考图内容已变化'):
        store.export()


def test_review_api_images_training_switch_and_cross_origin(dataset):
    from fastapi.testclient import TestClient
    pairs, rows, directory = dataset
    app = create_review_app(pairs, directory)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/assets/app.js').status_code == 200
        listing = client.get('/api/items', params={'search': 'sun gem'}).json()
        assert listing['total'] == 1
        ident = pair_id(rows[0])
        image = client.get(f'/api/images/{ident}/target')
        assert image.status_code == 200
        assert Image.open(io.BytesIO(image.content)).size == (16, 16)
        body = {'pair_ids': [ident], 'state': 'keep', 'train_selected': True, 'expected_revision': 0}
        assert client.post('/api/decisions', json=body, headers={'Origin': 'https://unrelated.example'}).status_code == 403
        assert client.post('/api/decisions', json=body).status_code == 200
        assert client.get('/api/summary').json()['train_selected_pairs'] == 1
        assert client.post('/api/decisions', json=body).status_code == 409
        response = client.post('/api/exports', json={})
        assert response.status_code == 200
        assert client.get(response.json()['download_url']).json()['selected_indices'] == [0, 2, 3]
        assert client.get('/api/exports/unknown/selection.json').status_code == 404


def test_reviewed_training_draws_only_selected_pairs_without_reindexing(dataset, monkeypatch):
    import importlib.util
    import torch
    import yaml
    from mcgen.config import Experiment
    from mcgen.smoke import tiny_config
    from mcgen.tasks import conditioned_prompts

    pairs, rows, directory = dataset
    root = pairs.parent
    data = tiny_config().to_dict()
    data['model'].update(conditioning='text_only', byte_tokens=False, text_dim=8, text_length=1,
                         visual_adapter=True, visual_dim=12, visual_grid=2, visual_layers=[0, 1])
    data['text'].update(backend='qwen3_vl_embedding', model='fake', revision='a'*40)
    data.update(conditioning_input='visual_tokens', task='joint')
    data['train'].update(trainable='visual_adapter', task_mix={'text': 0., 'reference': .2, 'joint': .8, 'unconditional': 0.})
    cfg = Experiment.from_dict(data)
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.to_dict()))
    (root/'data_audit.json').write_text(json.dumps({'pairs_sha256': file_hash(pairs), 'base_text_identity': 'test-text'}))
    cache = root/'visual_features'
    cache.mkdir()
    features = torch.randn(4, 4, 12).bfloat16()
    array = features.view(torch.uint16).numpy()
    array.tofile(cache/'features.uint16.mmap')
    (cache/'metadata.json').write_text(json.dumps({'complete': True, 'pairs_sha256': file_hash(pairs),
        'visual_identity': 'test-visual', 'shape': [4, 4, 12], 'chunks': {'0': hashlib.sha256(array.tobytes()).hexdigest()}}))
    text_cache = root/'text'
    text_cache.mkdir()
    prompts = conditioned_prompts(['sun gem', 'tin helmet'], cfg)
    (text_cache/'prompts.json').write_text(json.dumps(prompts))
    text_features = torch.randn(2, 1, 8).bfloat16()
    text_features.view(torch.uint16).numpy().tofile(text_cache/'features.uint16.mmap')
    np.ones((2, 1), dtype=np.uint8).tofile(text_cache/'masks.uint8.mmap')
    (text_cache/'metadata.json').write_text(json.dumps({'complete': True, 'text_identity': 'test-text',
        'prompts_sha256': file_hash(text_cache/'prompts.json'), 'shape': [2, 1, 8]}))
    store = ReviewStore(pairs, directory)
    store.update([pair_id(rows[0])], state='keep', train_selected=True)
    result = store.export()
    helper_path = Path(__file__).resolve().parents[1]/'scripts/prepare_reviewed_reference.py'
    helper_spec = importlib.util.spec_from_file_location('prepare_reviewed_test', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    out = root/'reviewed_run'
    before = file_hash(pairs)
    helper.prepare(Path(result['selection_path']), root, out, steps=10)
    assert not (root/'selection.json').exists()
    assert file_hash(pairs) == before
    assert (out/'visual_features').is_symlink()
    path = Path(__file__).resolve().parents[1]/'scripts/train_visual_reference.py'
    spec = importlib.util.spec_from_file_location('reviewed_training_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'TEXT_CACHE', str(text_cache))
    loaded = module.Data(out, partial=True)
    assert loaded.groups['train'] == [[0]]
    assert set(loaded.draw(100, torch.Generator().manual_seed(19))) == {0}
    assert torch.equal(loaded.batch([0], 'cpu')[3], features[:1])
    assert loaded.selection_sha256 == file_hash(out/'selection.json')
