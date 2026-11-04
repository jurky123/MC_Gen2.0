"""Spatial adaptation learns while the original generator remains bit-identical."""
import copy

import numpy as np
from PIL import Image
import pytest
import torch

from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.reference_adapter import initialize_visual_adapter, backbone_digest
from mcgen.runtime import load_checkpoint, save_checkpoint
from mcgen.sampling import sample
from mcgen.smoke import tiny_config
from mcgen.visual_tokens import visual_image


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    data = tiny_config().to_dict()
    data['model'].update(conditioning='text_only', byte_tokens=False, text_dim=8, text_length=1)
    data['text'].update(backend='qwen3_vl_embedding', model='fake', revision='a'*40)
    data['task'] = 'text'
    data['train']['task_mix'] = {'text': .9, 'reference': 0., 'joint': 0., 'unconditional': .1}
    base = Experiment.from_dict(data)
    source = PixelFlow(base.model)
    with torch.no_grad():
        source.output.weight.normal_(std=.1)
        for b in source.blocks:
            b.modulation[-1].bias.fill_(.3)
    source.eval().requires_grad_(False)
    new = copy.deepcopy(data)
    new['model'].update(visual_adapter=True, visual_dim=12, visual_grid=2, visual_layers=[0, 1])
    new['conditioning_input'] = 'visual_tokens'
    new['task'] = 'joint'
    new['train']['trainable'] = 'visual_adapter'
    new['train']['task_mix'] = {'text': 0., 'reference': .2, 'joint': .8, 'unconditional': 0.}
    cfg = Experiment.from_dict(new)
    state = {'config': base.to_dict(), 'model': source.state_dict()}
    model = initialize_visual_adapter(cfg, state, 'cpu')
    text = torch.randn(2, 1, 8)
    mask = torch.ones(2, 1, dtype=torch.bool)
    refs = torch.randn(2, 4, 12)
    return source, model, text, mask, refs, cfg


def test_visual_learning_and_exact_backbone_and_null_preservation():
    source, model, text, mask, refs, _ = setup()
    x, t = torch.randn(2, 4, 8, 8), torch.rand(2)
    before = backbone_digest(source)
    original = source(x, t, source.context(text, mask, None, None)).detach()
    empty = torch.zeros_like(mask)
    null = source(x, t, source.context(text, empty, None, None)).detach()
    with torch.no_grad():
        initial = model(x, t, model.context(text, mask, refs, None))
    assert torch.equal(original, initial)
    opt = torch.optim.AdamW(model.visual_adapter.parameters(), lr=.01)
    for _ in range(5):
        opt.zero_grad()
        model(x, t, model.context(text, mask, refs, None)).square().mean().backward()
        opt.step()
    assert model.visual_adapter.projection[-1].weight.grad.abs().sum() > 0
    assert model.visual_adapter.attention['0'].in_proj_weight.grad.abs().sum() > 0
    assert before == backbone_digest(model)
    assert torch.equal(original, model(x, t, model.context(text, mask, None, None)))
    assert torch.equal(null, model(x, t, model.context(text, empty, None, None)))
    assert not torch.equal(original, model(x, t, model.context(text, mask, refs, None)))
    assert all(not p.requires_grad and p.grad is None for n, p in model.named_parameters()
               if not n.startswith('visual_adapter.'))


def test_zero_reference_guidance_is_exact_text_and_image_only_ignores_text():
    source, model, text, mask, refs, _ = setup()
    with torch.no_grad():
        for gate in model.visual_adapter.gates.values():
            gate.fill_(.4)
    args = dict(seeds=[12, 13], steps=2, text_scale=2.)
    original = sample(source, text, mask, **args)
    assert torch.equal(original, sample(model, text, mask, refs, ref_scale=0, **args))
    a = sample(model, text, mask, refs, seeds=[12, 13], steps=2, text_scale=0)
    b = sample(model, text+100, mask, refs, seeds=[12, 13], steps=2, text_scale=0)
    assert torch.equal(a, b)
    assert not torch.equal(a, sample(model, text, mask, refs.flip(0), seeds=[12, 13], steps=2, text_scale=0))


def test_independent_cfg_matches_three_branch_formula():
    _, model, text, mask, refs, cfg = setup()
    with torch.no_grad():
        for gate in model.visual_adapter.gates.values():
            gate.fill_(.2)
        seeds = [1, 2]
        x = torch.cat([torch.randn(1, 4, cfg.model.size, cfg.model.size,
                        generator=torch.Generator().manual_seed(s)) for s in seeds])
        t = torch.ones(2)
        v0 = model(x, t, model.context(text, torch.zeros_like(mask), None, None))
        vt = model(x, t, model.context(text, mask, None, None))
        vtr = model(x, t, model.context(text, mask, refs, None))
        expected = (x-(v0+1.5*(vt-v0)+.7*(vtr-vt))).clamp(-1, 1)
        actual = sample(model, text, mask, refs, seeds=seeds, steps=1,
                        solver='euler', text_scale=1.5, ref_scale=.7)
    assert torch.equal(actual, expected)


def test_visual_checkpoint_reload_and_presence_contract(tmp_path):
    source, model, text, mask, refs, cfg = setup()
    with torch.no_grad():
        for gate in model.visual_adapter.gates.values():
            gate.fill_(.25)
    checkpoint = tmp_path/'visual.pt'
    save_checkpoint(checkpoint, {'config': cfg.to_dict(), 'model': model.state_dict()})
    state = load_checkpoint(checkpoint)
    loaded = PixelFlow(Experiment.from_dict(state['config']).model)
    loaded.load_state_dict(state['model'], strict=True)
    loaded.eval()
    assert backbone_digest(loaded) == backbone_digest(model)
    args = dict(seeds=[1, 2], steps=2)
    assert torch.equal(sample(source, text, mask, **args), sample(loaded, text, mask, **args))
    assert torch.equal(sample(model, text, mask, refs, **args), sample(loaded, text, mask, refs, **args))
    assert len(loaded.context(text, mask, refs, torch.zeros(2, dtype=torch.bool))) == 2
    with pytest.raises(ValueError, match='contract'):
        loaded.context(text, mask, refs[:, :2], None)
    with pytest.raises(ValueError, match='presence'):
        loaded.context(text, mask, refs, torch.ones(1, dtype=torch.bool))


def test_visual_config_requires_frozen_compatible_backbone():
    _, _, _, _, _, cfg = setup()
    for key, value in [('trainable', 'all')]:
        data = cfg.to_dict()
        data['train'][key] = value
        with pytest.raises(ValueError, match='frozen-backbone'):
            Experiment.from_dict(data)
    data = cfg.to_dict()
    data['model']['visual_layers'] = [cfg.model.depth]
    with pytest.raises(ValueError, match='outside'):
        Experiment.from_dict(data)


def test_visual_preprocessing_preserves_whole_image_and_alpha():
    im = Image.new('RGBA', (80, 40), (255, 0, 0, 255))
    result = np.array(visual_image(im))
    assert result.shape == (256, 256, 3)
    assert (result[0] == 255).all()
    assert (result[128, 0] == [255, 0, 0]).all()
    transparent = visual_image(Image.new('RGBA', (20, 20), (0, 0, 0, 0)))
    assert (np.array(transparent) == 255).all()


def test_public_runtime_routes_separate_text_and_visual_conditions(tmp_path, monkeypatch):
    from mcgen.runtime import Runtime
    from mcgen.images import decode
    source, model, text, mask, refs, cfg = setup()
    with torch.no_grad():
        for gate in model.visual_adapter.gates.values():
            gate.fill_(.2)

    class Text:
        identity = 'test-text'

        def __init__(self, *args):
            pass

        @torch.inference_mode()
        def encode(self, prompts, device):
            return text[:len(prompts)].clone().to(device), mask[:len(prompts)].clone().to(device)

    class Visual:
        identity = 'test-vision'

        def __init__(self, *args):
            pass

        @torch.inference_mode()
        def encode(self, images, device):
            return refs[:len(images)].clone().to(device)

    monkeypatch.setattr('mcgen.runtime.TextEncoder', Text)
    monkeypatch.setattr('mcgen.visual_tokens.VisualTokenEncoder', Visual)
    assert Text().encode(['helmet'], 'cpu')[1].is_inference()
    checkpoint = tmp_path/'runtime.pt'
    save_checkpoint(checkpoint, {'config': cfg.to_dict(), 'model': model.state_dict(), 'step': 5,
        'text_identity': Text.identity, 'visual_identity': Visual.identity, 'catalog_fingerprint': 'test-pairs'})
    runtime = Runtime(checkpoint)
    images = [Image.new('RGB', (32, 32), 'red'), Image.new('RGB', (32, 32), 'blue')]
    actual = runtime.generate(['helmet', 'helmet'], images, seeds=[1, 2], steps=2)
    expected = sample(model, text, mask, refs, seeds=[1, 2], steps=2)
    assert all(np.array_equal(a, decode(b)) for a, b in zip(actual, expected))
    plain = runtime.generate(['helmet', 'helmet'], seeds=[1, 2], steps=2)
    original = sample(source, text, mask, seeds=[1, 2], steps=2)
    assert all(np.array_equal(a, decode(b)) for a, b in zip(plain, original))
    disabled = runtime.generate(['helmet', 'helmet'], images, seeds=[1, 2], steps=2, ref_scale=0)
    assert all(np.array_equal(a, b) for a, b in zip(disabled, plain))
    with pytest.raises(ValueError, match='requires a reference'):
        runtime.generate(['helmet'], conditioning_mode='visual_tokens')
