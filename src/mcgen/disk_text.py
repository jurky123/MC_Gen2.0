"""Exact full-length frozen BF16 features; no padding trimming or quantization."""
import json
from pathlib import Path

import numpy as np
import torch

from .catalog import file_hash


class DiskTextEncoder:
    def __init__(self, directory, expected_identity=None):
        root = Path(directory)
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
