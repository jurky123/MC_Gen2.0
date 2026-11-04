"""Frozen Qwen vision tower, without the language model or global pooling.

The official 2x2 patch merger preserves an 8x8 spatial grid at 256px input.
Singleton forwards are canonical so offline caches and public inference agree.
"""
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from .catalog import file_hash
from .config import fingerprint
from .qwen_embedding import embedding_image

IMAGE_SIZE = 256
TOKEN_GRID = 8
FEATURE_DIM = 4096


def visual_image(value):
    from PIL import Image, ImageOps
    im = ImageOps.contain(embedding_image(value), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 'white')
    canvas.paste(im, ((IMAGE_SIZE-im.width)//2, (IMAGE_SIZE-im.height)//2))
    return canvas


class VisualTokenEncoder:
    def __init__(self, text_config, device='cpu'):
        from safetensors import safe_open
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        root = Path(os.environ.get('MCGEN_EMBEDDING_MODEL') or text_config.model)
        manifest = json.loads((root/'verified_download.json').read_text())
        if (not manifest['complete'] or manifest['repo'] != 'Qwen/Qwen3-VL-Embedding-8B'
                or manifest['revision'] != text_config.revision):
            raise ValueError('Wrong or unverified visual encoder')
        files = {}
        for row in manifest['files']:
            digest = file_hash(root/row['name'])
            if digest != row['verified_sha256']:
                raise ValueError('Visual encoder artifacts changed: '+row['name'])
            files[row['name']] = digest
        config = json.loads((root/'config.json').read_text())['vision_config']
        self.identity = fingerprint({'protocol': 'qwen3_vl_spatial_merged_8x8_v1',
            'revision': text_config.revision, 'files': files, 'vision_config': config,
            'image_size': IMAGE_SIZE, 'resize': 'contain_lanczos_center_white',
            'batch_policy': 'singleton', 'dtype': 'bfloat16',
            'output': [TOKEN_GRID**2, FEATURE_DIM], 'features': 'pooler_output_no_global_pool'})
        self.device = device
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True).image_processor
        cfg = Qwen3VLVisionConfig(**config)
        cfg._attn_implementation = 'sdpa'
        self.model = Qwen3VLVisionModel(cfg).to(dtype=torch.bfloat16)
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        shards = defaultdict(list)
        prefix = 'model.visual.'
        for name, shard in index.items():
            if name.startswith(prefix):
                shards[shard].append(name)
        state = {}
        for shard, names in shards.items():
            with safe_open(root/shard, framework='pt', device='cpu') as f:
                for name in names:
                    state[name[len(prefix):]] = f.get_tensor(name)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images, output_device='cpu'):
        if not images or any(im is None for im in images):
            raise ValueError('one real reference required per visual encoding')
        values = []
        for im in images:
            batch = self.processor(images=[visual_image(im)], do_resize=False, return_tensors='pt')
            if batch['image_grid_thw'].tolist() != [[1, 16, 16]]:
                raise ValueError('Unexpected Qwen spatial grid')
            result = self.model(batch['pixel_values'].to(self.device, dtype=torch.bfloat16),
                                grid_thw=batch['image_grid_thw'].to(self.device))
            h = result.pooler_output
            if h.shape != (TOKEN_GRID**2, FEATURE_DIM) or not torch.isfinite(h).all():
                raise ValueError('Invalid spatial visual features')
            values.append(h.to(output_device))
        return torch.stack(values)
