import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcgen.config import Experiment
from mcgen.model import PixelFlow
from mcgen.runtime import Runtime, load_checkpoint
from mcgen.smoke import tiny_config, fixture
from mcgen.training import Trainer


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_attribute_path_is_invariant_to_pixel_permutation_but_retains_color():
    cfg = tiny_config().model
    cfg.reference_encoding = "dual"
    model = PixelFlow(cfg)
    text = torch.randn(2, 4, cfg.text_dim)
    mask = torch.ones(2, 4, dtype=torch.bool)
    ref = torch.randn(2, 4, cfg.ref_size, cfg.ref_size)
    present = torch.ones(2, dtype=torch.bool)
    permuted = ref.flatten(2)[:, :, torch.randperm(cfg.ref_size ** 2)].reshape_as(ref)
    a = model.context(text, mask, ref, present)
    b = model.context(text, mask, permuted, present)
    for x, y in zip(a[0], b[0]):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    changed = model.context(text, mask, ref + .5, present)
    assert not torch.allclose(a[0][2], changed[0][2])
    shape = model.context(text, mask, ref, present, "shape")
    rotated = model.context(text, mask, ref.rot90(1, (-2, -1)), present, "shape")
    assert not torch.allclose(shape[0][1], rotated[0][1])
    assert not a[1][1][:, 1:].any()
    assert shape[1][1][:, 1:].all()


def test_old_config_and_weights_remain_strict_and_modes_are_validated():
    cfg = tiny_config().to_dict()
    cfg["model"].pop("reference_encoding")
    cfg["train"].pop("trainable")
    old = Experiment.from_dict(cfg)
    assert old.model.reference_encoding == "spatial"
    model = PixelFlow(old.model)
    text = torch.randn(1, 4, old.model.text_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="dual"):
        model.context(text, mask, None, None, "attributes")
    with pytest.raises(ValueError, match="per sample"):
        model.context(text, mask, None, None, ["shape", "shape"])


def test_adapter_transfer_freeze_resume_and_runtime_routing(tmp_path):
    catalog = fixture(tmp_path / "fixture")
    cfg = tiny_config()
    base = Trainer(cfg, catalog, tmp_path / "base")
    checkpoint = base.run(stop_after=2)
    cfg = copy.deepcopy(cfg)
    cfg.model.reference_encoding = "dual"
    cfg.train.trainable = "attributes"
    dual = Trainer(cfg, catalog, tmp_path / "dual", initialize=checkpoint)
    initial = {n: p.clone() for n, p in dual.model.state_dict().items()}
    x = torch.randn(2, 4, cfg.model.size, cfg.model.size)
    text, mask = base.encoder.encode(["green bottle", "blue bottle"], "cpu")
    ref = torch.randn(2, 4, cfg.model.ref_size, cfg.model.ref_size)
    present = torch.ones(2, dtype=torch.bool)
    t = torch.tensor([.3, .8])
    old = base.model(x, t, base.model.context(text, mask, ref, present))
    new = dual.model(x, t, dual.model.context(text, mask, ref, present, "shape"))
    torch.testing.assert_close(old, new, atol=0, rtol=0)
    dual.run(stop_after=3)
    for name, value in dual.model.state_dict().items():
        if not dual.is_attribute(name):
            assert torch.equal(value, initial[name]), name
    assert any(not torch.equal(value, initial[name]) for name, value in dual.model.state_dict().items()
               if dual.is_attribute(name))
    resumed = Trainer(cfg, catalog, tmp_path / "dual", resume=tmp_path / "dual/latest.pt")
    dual.run(); resumed.run()
    for n, value in dual.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[n]), n
    runtime = Runtime(tmp_path / "dual/latest.pt")
    refs = [np.random.default_rng(7).integers(0, 256, (16, 16, 4), dtype=np.uint8)]
    auto = runtime.generate(["green bottle"], refs, seeds=[1], steps=2)
    explicit = runtime.generate(["green bottle"], refs, seeds=[1], steps=2, reference_mode="attributes")
    assert np.array_equal(auto[0], explicit[0])
    auto = runtime.generate([""], refs, seeds=[1], steps=2)
    explicit = runtime.generate([""], refs, seeds=[1], steps=2, reference_mode="shape")
    assert np.array_equal(auto[0], explicit[0])
    assert load_checkpoint(tmp_path / "dual/latest.pt")["initialization"]["step"] == 2
    with pytest.raises(ValueError, match="mutually exclusive"):
        Trainer(cfg, catalog, tmp_path / "bad", resume=checkpoint, initialize=checkpoint)
