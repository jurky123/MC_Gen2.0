"""Exact full-length frozen BF16 features; no padding trimming or quantization."""
import json
from pathlib import Path

import numpy as np
import torch

from .catalog import file_hash


class DiskTextEncoder:
    def __init__(self, directory, expected_identity=None):
        root = Path(directory)
        self.root = root.resolve()
        meta = json.loads((root / 'metadata.json').read_text())
        if not meta['complete']:
            raise ValueError('Incomplete frozen text cache')
        if expected_identity is not None and meta['text_identity'] != expected_identity:
            raise ValueError('Frozen text identity mismatch')
        if file_hash(root / 'prompts.json') != meta['prompts_sha256']:
            raise ValueError('Frozen prompt index changed')
        prompts = json.loads((root / 'prompts.json').read_text())
        self.index = {p: i for i, p in enumerate(prompts)}
        if len(self.index) != len(prompts): raise ValueError('Duplicate cache prompts')
        shape = tuple(meta['shape'])
        if shape[0] != len(prompts): raise ValueError('Prompt count mismatch')
        if (root/'features.uint16.mmap').stat().st_size != int(np.prod(shape))*2:
            raise ValueError('Feature cache size mismatch')
        if (root/'masks.uint8.mmap').stat().st_size != shape[0]*shape[1]:
            raise ValueError('Mask cache size mismatch')
        self.features = np.memmap(root/'features.uint16.mmap', mode='r', dtype='uint16', shape=shape)
        self.masks = np.memmap(root/'masks.uint8.mmap', mode='r', dtype='uint8', shape=shape[:2])
        self.identity = meta['text_identity']
        self.hits = 0

    def encode(self, prompts, output_device):
        ids = [self.index[p] for p in prompts]  # Missing prompts fail closed; never silently substitute.
        self.hits += len(ids)
        features = torch.from_numpy(np.array(self.features[ids], copy=True)).view(torch.bfloat16)
        masks = torch.from_numpy(np.array(self.masks[ids], copy=True)).bool()
        return features.to(output_device), masks.to(output_device)

    def cache_stats(self):
        return {'hits': self.hits, 'misses': 0, 'entries': len(self.index), 'backend': 'exact_bfloat16_mmap'}


class RaggedDiskTextEncoder:
    """Lossless BF16 token sequences; pad only the requested batch on read."""
    is_ragged = True

    def __init__(self,directory,expected_identity=None):
        root=Path(directory);self.root=root.resolve()
        meta=json.loads((root/'metadata.json').read_text())
        if not meta['complete'] or meta.get('format') != 'ragged_bfloat16_text_v1':
            raise ValueError('Incomplete or unsupported token text cache')
        if expected_identity is not None and meta['text_identity'] != expected_identity:
            raise ValueError('Frozen text identity mismatch')
        for name,key in [('prompts.json','prompts_sha256'),('offsets.uint64.mmap','offsets_sha256'),
                         ('features.uint16.mmap','features_sha256')]:
            if file_hash(root/name) != meta[key]:
                raise ValueError('Frozen token cache changed: '+name)
        prompts=json.loads((root/'prompts.json').read_text())
        self.index={p:i for i,p in enumerate(prompts)}
        if len(self.index)!=len(prompts):raise ValueError('Duplicate token cache prompts')
        shape=tuple(meta['shape'])
        if len(shape)!=2 or shape[1]!=4096 or (root/'features.uint16.mmap').stat().st_size!=int(np.prod(shape))*2:
            raise ValueError('Token feature cache shape mismatch')
        self.offsets=np.memmap(root/'offsets.uint64.mmap',dtype='uint64',mode='r',shape=(len(prompts)+1,))
        if self.offsets[0]!=0 or self.offsets[-1]!=shape[0] or np.any(self.offsets[1:]<self.offsets[:-1]):
            raise ValueError('Invalid packed token offsets')
        if int(np.diff(self.offsets).max())>meta['max_tokens']:
            raise ValueError('Token cache exceeds the declared token limit')
        self.features=np.memmap(root/'features.uint16.mmap',dtype='uint16',mode='r',shape=shape)
        self.identity=meta['text_identity'];self.hits=0

    def encode(self,prompts,output_device):
        if not prompts:raise ValueError('Empty token batch')
        ids=[self.index[p] for p in prompts]
        spans=[(int(self.offsets[i]),int(self.offsets[i+1])) for i in ids]
        length=max(1,max(b-a for a,b in spans))
        values=torch.zeros(len(ids),length,4096,dtype=torch.bfloat16)
        masks=torch.zeros(len(ids),length,dtype=torch.bool)
        for i,(a,b) in enumerate(spans):
            values[i,:b-a]=torch.from_numpy(np.array(self.features[a:b],copy=True)).view(torch.bfloat16)
            masks[i,:b-a]=True
        self.hits+=len(ids)
        return values.to(output_device),masks.to(output_device)

    def cache_stats(self):
        return {'hits':self.hits,'misses':0,'entries':len(self.index),'backend':'ragged_bfloat16_text_v1'}


def load_text_cache(directory,expected_identity=None):
    meta=json.loads((Path(directory)/'metadata.json').read_text())
    cls=RaggedDiskTextEncoder if meta.get('format')=='ragged_bfloat16_text_v1' else DiskTextEncoder
    return cls(directory,expected_identity)


class MergedDiskTextEncoder:
    """Route exact prompt keys across compatible immutable caches, preserving batch order."""

    def __init__(self, encoders):
        if not encoders:
            raise ValueError('At least one text cache is required')
        self.encoders = encoders
        self.identity = encoders[0].identity
        self.index = {}
        shape = encoders[0].features.shape[1:]
        for i, encoder in enumerate(encoders):
            if encoder.identity != self.identity or encoder.features.shape[1:] != shape:
                raise ValueError('Incompatible frozen text caches')
            for prompt in encoder.index:
                self.index.setdefault(prompt, i)

    def encode(self, prompts, output_device):
        if not prompts:
            raise ValueError('empty prompt batch')
        groups = {}
        for position, prompt in enumerate(prompts):
            groups.setdefault(self.index[prompt], []).append((position, prompt))
        features = masks = None
        for cache, batch in groups.items():
            x, m = self.encoders[cache].encode([p for _, p in batch], output_device)
            if features is None:
                features = x.new_empty((len(prompts), *x.shape[1:]))
                masks = m.new_empty((len(prompts), *m.shape[1:]))
            positions = [i for i, _ in batch]
            features[positions], masks[positions] = x, m
        return features, masks

    def cache_stats(self):
        return {'backend': 'merged_exact_bfloat16', 'entries': len(self.index),
                'hits': sum(e.hits for e in self.encoders), 'misses': 0}
