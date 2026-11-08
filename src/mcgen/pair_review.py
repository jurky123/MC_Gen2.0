"""Local, persistent human review of original-HD/MC training pairs."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps

from .catalog import atomic_json, file_hash
from .pair_selection import FORMAT, pair_id, selected_indices

STATES = ('pending', 'keep', 'reject', 'unsure')
KINDS = ('sword', 'pickaxe', 'axe', 'hoe', 'shovel', 'helmet', 'chestplate', 'leggings',
         'boots', 'shield', 'bow', 'crossbow', 'gem', 'crystal', 'ingot', 'nugget', 'dust',
         'gear', 'potion', 'bottle', 'apple', 'door')
DEFAULT = {'state': 'pending', 'train_selected': False, 'reason': '', 'note': '', 'updated_at': 0.}


class ReviewConflict(ValueError):
    pass


def category(prompt):
    return next((word for word in reversed(prompt.lower().split()) if word in KINDS), 'other')


def target_hash(array):
    array = array.copy()
    array[array[..., 3] == 0, :3] = 0
    return hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()


class ReviewStore:
    def __init__(self, pairs_file, directory):
        self.pairs_file = Path(pairs_file).resolve()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = file_hash(self.pairs_file)
        self.rows = json.loads(self.pairs_file.read_text())
        self.pairs, self.groups = {}, {}
        for i, row in enumerate(self.rows):
            if row.get('feature_index') != i or row['split'] not in ('train', 'val'):
                raise ValueError('Expected original canonical train/val pair indices')
            ident = pair_id(row)
            if ident in self.pairs:
                raise ValueError('Duplicate pair identity')
            self.pairs[ident] = {**row, 'pair_id': ident, 'index': i}
            group = self.groups.setdefault(row['target_hash'], {'target_hash': row['target_hash'],
                'prompt': row['prompt'], 'filename': row.get('source_filename', ''),
                'split': row['split'], 'category': category(row['prompt']), 'pair_ids': []})
            if group['split'] != row['split']:
                raise ValueError('Target split leakage')
            group['pair_ids'].append(ident)
        for group in self.groups.values():
            group['pair_ids'].sort(key=lambda ident: (self.pairs[ident]['reference_generator'], ident))
        self.order = sorted(self.groups, key=lambda h: (self.groups[h]['prompt'].casefold(), h))
        self.database = self.directory/'review.sqlite3'
        self.lock = threading.RLock()
        self.maps = {}
        with self.connection() as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript('''
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(
                    pair_id TEXT PRIMARY KEY,state TEXT NOT NULL,train_selected INTEGER NOT NULL,
                    reason TEXT NOT NULL,note TEXT NOT NULL,updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL,created_at REAL NOT NULL,undone INTEGER NOT NULL DEFAULT 0);
            ''')
            old = con.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            if old is not None and old[0] != self.fingerprint:
                raise ValueError('Review database belongs to a different pair manifest; use a new review directory')
            con.execute("INSERT OR IGNORE INTO meta VALUES('fingerprint',?)", (self.fingerprint,))
            con.execute("INSERT OR IGNORE INTO meta VALUES('revision','0')")

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def snapshot(self):
        with self.lock, self.connection() as con:
            con.execute('BEGIN')
            decisions = {r['pair_id']: {**dict(r), 'train_selected': bool(r['train_selected'])}
                         for r in con.execute('SELECT * FROM decisions')}
            revision = int(con.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
            con.commit()
        return decisions, revision

    def summary(self):
        decisions, revision = self.snapshot()
        counts = {s: Counter({k: 0 for k in STATES}) for s in ('train', 'val')}
        chosen = set()
        selected_pairs = 0
        for ident, row in self.pairs.items():
            d = decisions.get(ident, DEFAULT)
            counts[row['split']][d['state']] += 1
            if d['train_selected']:
                selected_pairs += 1
                chosen.add(row['target_hash'])
        return {'revision': revision, 'pairs': len(self.pairs), 'targets': len(self.groups),
                'counts': counts, 'train_selected_pairs': selected_pairs, 'train_selected_targets': len(chosen),
                'categories': sorted({g['category'] for g in self.groups.values()}),
                'dataset_name': 'MC 高清参考图配对集', 'fingerprint': self.fingerprint}

    def item(self, target, decisions=None):
        group = self.groups[target]
        decisions = self.snapshot()[0] if decisions is None else decisions
        variants = []
        for ident in group['pair_ids']:
            row = self.pairs[ident]
            variants.append({'pair_id': ident, 'generator': row['reference_generator'],
                'reference_url': f'/api/images/{ident}/reference',
                'target_url': f'/api/images/{ident}/target',
                **{k: decisions.get(ident, DEFAULT)[k] for k in DEFAULT}})
        return {**group, 'pairs': variants,
                'target_url': variants[0]['target_url'], 'native_size': self.pairs[group['pair_ids'][0]]['target']['shape'][1]}

    def listing(self, split='train', state='pending', search='', kind='', selected='all', page=1, limit=24):
        decisions, revision = self.snapshot()
        words = search.casefold().split()
        result = []
        for key in self.order:
            g = self.groups[key]
            if split != 'all' and g['split'] != split:
                continue
            if kind and g['category'] != kind:
                continue
            text = (g['prompt']+' '+g['filename']).casefold()
            if not all(word in text for word in words):
                continue
            matching = []
            for ident in g['pair_ids']:
                d = decisions.get(ident, DEFAULT)
                if state != 'all' and d['state'] != state:
                    continue
                if selected != 'all' and d['train_selected'] != (selected == 'yes'):
                    continue
                matching.append(ident)
            if matching:
                result.append(key)
        start = (page-1)*limit
        return {'revision': revision, 'total': len(result), 'page': page, 'limit': limit,
                'items': [self.item(k, decisions) for k in result[start:start+limit]]}

    def update(self, ids, state=None, train_selected=None, reason=None, note=None, expected_revision=None):
        ids = list(dict.fromkeys(ids))
        if not ids or len(ids) > 200 or any(ident not in self.pairs for ident in ids):
            raise ValueError('Unknown or invalid pair selection')
        if state is not None and state not in STATES:
            raise ValueError('Unknown review state')
        with self.lock, self.connection() as con:
            con.execute('BEGIN IMMEDIATE')
            try:
                revision = int(con.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
                if expected_revision is not None and revision != expected_revision:
                    raise ReviewConflict('另一个页面已更新筛选记录，请刷新后重试。')
                before, after = {}, {}
                for ident in ids:
                    row = con.execute('SELECT * FROM decisions WHERE pair_id=?', (ident,)).fetchone()
                    old = dict(row) if row is not None else None
                    current = dict(old or DEFAULT)
                    next_state = state if state is not None else current['state']
                    chosen = bool(current['train_selected']) if train_selected is None else train_selected
                    if next_state != 'keep':
                        chosen = False
                    if chosen and self.pairs[ident]['split'] != 'train':
                        raise ValueError('验证配对不能加入训练；原始划分保持不变。')
                    if train_selected and next_state != 'keep':
                        raise ValueError('请先确认配对可用，或使用“选入训练”。')
                    value = {'pair_id': ident, 'state': next_state, 'train_selected': int(chosen),
                        'reason': reason if reason is not None else (current['reason'] if next_state == current['state'] else ''),
                        'note': current['note'] if note is None else note, 'updated_at': time.time()}
                    if all(current.get(k) == value[k] for k in ('state', 'train_selected', 'reason', 'note')):
                        continue
                    before[ident], after[ident] = old, value
                    con.execute('INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?)', tuple(value.values()))
                if after:
                    con.execute('INSERT INTO events(before_json,after_json,created_at) VALUES(?,?,?)',
                                (json.dumps(before), json.dumps(after), time.time()))
                    revision += 1
                    con.execute("UPDATE meta SET value=? WHERE key='revision'", (str(revision),))
                con.commit()
                return revision
            except Exception:
                con.rollback()
                raise

    def undo(self, expected_revision=None):
        with self.lock, self.connection() as con:
            con.execute('BEGIN IMMEDIATE')
            try:
                revision = int(con.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
                if expected_revision is not None and revision != expected_revision:
                    raise ReviewConflict('筛选记录已变化，请刷新后重试。')
                event = con.execute('SELECT * FROM events WHERE undone=0 ORDER BY id DESC LIMIT 1').fetchone()
                if event is None:
                    con.rollback()
                    return revision
                for ident, old in json.loads(event['before_json']).items():
                    if old is None:
                        con.execute('DELETE FROM decisions WHERE pair_id=?', (ident,))
                    else:
                        con.execute('INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?)',
                            tuple(old[k] for k in ('pair_id', 'state', 'train_selected', 'reason', 'note', 'updated_at')))
                con.execute('UPDATE events SET undone=1 WHERE id=?', (event['id'],))
                revision += 1
                con.execute("UPDATE meta SET value=? WHERE key='revision'", (str(revision),))
                con.commit()
                return revision
            except Exception:
                con.rollback()
                raise

    def read_target(self, row):
        spec = row['target']
        with self.lock:
            if spec['path'] not in self.maps:
                self.maps[spec['path']] = np.memmap(spec['path'], mode='r', dtype='uint8', shape=tuple(spec['shape']))
            value = np.array(self.maps[spec['path']][spec['index']])
        if target_hash(value) != row['target_hash']:
            raise ValueError('MC 原图内容已变化，请重新检查数据。')
        return value

    @lru_cache(maxsize=256)
    def image(self, ident, kind, size):
        row = self.pairs[ident]
        if kind == 'target':
            im = Image.fromarray(self.read_target(row))
        else:
            if file_hash(row['reference_file']) != row['reference_sha256']:
                raise ValueError('高清参考图内容已变化。')
            with Image.open(row['reference_file']) as original:
                im = ImageOps.exif_transpose(original).convert('RGBA')
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        im.save(output, format='PNG')
        return output.getvalue()

    def export(self, validation_policy='exclude_rejected'):
        if validation_policy not in ('exclude_rejected', 'approved_only'):
            raise ValueError('Unknown validation policy')
        decisions, revision = self.snapshot()
        if file_hash(self.pairs_file) != self.fingerprint:
            raise ValueError('源配对清单已变化，请重新打开筛选服务。')
        chosen = []
        for ident, row in self.pairs.items():
            d = decisions.get(ident, DEFAULT)
            include = d['state'] == 'keep' and d['train_selected'] if row['split'] == 'train' else (
                d['state'] != 'reject' if validation_policy == 'exclude_rejected' else d['state'] == 'keep')
            if include:
                chosen.append(row['index'])
        chosen.sort()
        if not any(self.rows[i]['split'] == 'train' for i in chosen):
            raise ValueError('还没有选入训练的配对。请勾选“用于本次训练”。')
        if not any(self.rows[i]['split'] == 'val' for i in chosen):
            raise ValueError('没有可用验证配对，请调整验证集导出范围。')
        checked_targets = set()
        for i in chosen:
            row = self.rows[i]
            if file_hash(row['reference_file']) != row['reference_sha256']:
                raise ValueError('有参考图内容已变化，暂不能导出。')
            if row['target_hash'] not in checked_targets:
                self.read_target(row)
                checked_targets.add(row['target_hash'])
        counts = Counter(self.rows[i]['split'] for i in chosen)
        targets = {s: len({self.rows[i]['target_hash'] for i in chosen if self.rows[i]['split'] == s}) for s in ('train', 'val')}
        selection = {'format': FORMAT, 'source_pairs_sha256': self.fingerprint,
            'created_at': time.time(), 'review_revision': revision, 'validation_policy': validation_policy,
            'selected_indices': chosen, 'selected_pair_ids': [pair_id(self.rows[i]) for i in chosen],
            'counts': dict(counts), 'unique_targets': targets,
            'training_rule': 'only explicitly train_selected and keep; original cache indices and splits retained'}
        ident = uuid.uuid4().hex
        out = self.directory/'exports'/ident
        out.mkdir(parents=True)
        atomic_json(out/'selection.json', selection)
        selected_indices(out/'selection.json', self.pairs_file, self.rows)
        with (out/'selected_pairs.jsonl').open('w') as f:
            for i in chosen:
                f.write(json.dumps(self.rows[i], ensure_ascii=False)+'\n')
        summary = {'id': ident, 'train_pairs': counts['train'], 'train_targets': targets['train'],
            'val_pairs': counts['val'], 'val_targets': targets['val'], 'revision': revision,
            'selection_path': str(out/'selection.json'), 'download_url': f'/api/exports/{ident}/selection.json',
            'archive_url': f'/api/exports/{ident}/selection.zip'}
        atomic_json(out/'summary.json', summary)
        with zipfile.ZipFile(out/'selection.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ('selection.json', 'selected_pairs.jsonl', 'summary.json'):
                z.write(out/name, name)
        return summary


def create_review_app(pairs_file, directory):
    from fastapi import FastAPI, HTTPException, Request, Query
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    from typing import Literal

    store = ReviewStore(pairs_file, directory)
    app = FastAPI(title='MC Pair Studio', docs_url=None, redoc_url=None)
    app.state.review = store
    assets = Path(__file__).with_name('review_ui')

    class Change(BaseModel):
        pair_ids: list[str] = Field(min_length=1, max_length=200)
        state: Literal['pending', 'keep', 'reject', 'unsure'] | None = None
        train_selected: bool | None = None
        reason: str | None = Field(default=None, max_length=200)
        note: str | None = Field(default=None, max_length=2000)
        expected_revision: int | None = Field(default=None, ge=0)

    class Revision(BaseModel):
        expected_revision: int | None = Field(default=None, ge=0)

    class Export(BaseModel):
        validation_policy: Literal['exclude_rejected', 'approved_only'] = 'exclude_rejected'

    @app.middleware('http')
    async def same_origin(request: Request, call_next):
        # Local writes are not available to unrelated websites in the browser.
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if request.headers.get('sec-fetch-site') == 'cross-site' or (origin and origin != str(request.base_url).rstrip('/')):
                return Response('Cross-origin writes are not allowed', status_code=403)
        return await call_next(request)

    def run(action):
        try:
            return action()
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/')
    def index():
        return FileResponse(assets/'index.html')

    @app.get('/api/summary')
    def summary():
        return store.summary()

    @app.get('/api/items')
    def items(split: Literal['train', 'val', 'all'] = 'train', state: Literal['pending', 'keep', 'reject', 'unsure', 'all'] = 'pending',
              search: str = Query(default='', max_length=200), category: str = '', selected: Literal['all', 'yes', 'no'] = 'all',
              page: int = Query(default=1, ge=1), limit: int = Query(default=24, ge=1, le=48)):
        return store.listing(split, state, search, category, selected, page, limit)

    @app.get('/api/items/{target}')
    def item(target: str):
        if target not in store.groups:
            raise HTTPException(404, 'Unknown item')
        return store.item(target)

    @app.get('/api/images/{ident}/{kind}')
    def image(ident: str, kind: Literal['reference', 'target'], size: int = Query(default=640, ge=64, le=2048)):
        if ident not in store.pairs:
            raise HTTPException(404, 'Unknown pair')
        return run(lambda: Response(store.image(ident, kind, size), media_type='image/png',
                                    headers={'Cache-Control': 'private, max-age=3600'}))

    @app.post('/api/decisions')
    def decision(change: Change):
        return run(lambda: {'revision': store.update(change.pair_ids, change.state, change.train_selected,
                            change.reason, change.note, change.expected_revision)})

    @app.post('/api/undo')
    def undo(revision: Revision):
        return run(lambda: {'revision': store.undo(revision.expected_revision)})

    @app.post('/api/exports')
    def export(options: Export):
        return run(lambda: store.export(options.validation_policy))

    @app.get('/api/exports/{ident}/{name}')
    def download(ident: str, name: str):
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident) or name not in ('selection.json', 'selected_pairs.jsonl', 'summary.json', 'selection.zip'):
            raise HTTPException(404, 'Unknown export')
        path = store.directory/'exports'/ident/name
        if not path.is_file():
            raise HTTPException(404, 'Export not found')
        return FileResponse(path, filename=name)

    app.mount('/assets', StaticFiles(directory=assets), name='assets')
    return app
