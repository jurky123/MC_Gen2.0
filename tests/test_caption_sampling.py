"""Multiple captions must not increase an image's chance of being trained on."""
from collections import Counter
from types import SimpleNamespace
import torch
from mcgen.tasks import TaskSampler


def test_caption_choice_does_not_double_image_weight():
    dataset = SimpleNamespace(records=[
        {'target_hash': 'one-caption', 'prompt': 'a gem'},
        {'target_hash': 'two-captions', 'prompt': 'sun gem'},
        {'target_hash': 'two-captions', 'prompt': 'A sun gem with a yellow body.'},
    ])
    config = SimpleNamespace(task='text', train=SimpleNamespace(
        task_mix={'text': 1., 'reference': 0., 'joint': 0., 'unconditional': 0.}))
    sampler = TaskSampler(dataset, config)
    indices, modes = sampler.sample(40000, torch.Generator().manual_seed(112))
    counts = Counter(indices)
    # Row-uniform sampling would instead allocate 2/3 to the two-caption image.
    assert abs(counts[0] / len(indices) - .5) < .015
    assert abs(counts[1] / (counts[1] + counts[2]) - .5) < .015
    assert set(modes) == {'text'}
    repeated, _ = sampler.sample(40000, torch.Generator().manual_seed(112))
    assert indices == repeated
