import base64
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcgen.api import create_app
from mcgen.config import Experiment
from mcgen.runtime import Runtime, load_checkpoint
from mcgen.smoke import fixture, tiny_config
from mcgen.training import Trainer, flow_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_resume_matches_uninterrupted_and_rejects_config_changes(tmp_path):
    catalog = fixture(tmp_path / "fixture")
    cfg = tiny_config()
    full = Trainer(cfg, catalog, tmp_path / "full")
    full.run()
    half = Trainer(cfg, catalog, tmp_path / "half")
    half.run(stop_after=3)
    resumed = Trainer(cfg, catalog, tmp_path / "half", resume=tmp_path / "half/latest.pt")
    resumed.run()
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    altered = copy.deepcopy(cfg)
    altered.train.lr *= 2
    with pytest.raises(ValueError, match="exactly matching"):
        Trainer(altered, catalog, tmp_path / "other", resume=tmp_path / "half/latest.pt")


def test_api_and_runtime_match_for_both_tasks(tmp_path):
    from fastapi.testclient import TestClient
    catalog = fixture(tmp_path / "fixture")
    checkpoint = Trainer(tiny_config(), catalog, tmp_path / "train").run(stop_after=2)
    runtime = Runtime(checkpoint)
    ref = np.zeros((16, 16, 4), np.uint8)
    ref[3:12, 4:10] = [255, 0, 0, 255]
    buf = io.BytesIO()
    Image.fromarray(ref).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    with TestClient(create_app(checkpoint)) as client:
        assert client.get("/health").json()["ready"]
        for image in (None, ref):
            request = {"prompt": "red bottle", "seed": 7, "steps": 2}
            if image is not None:
                request["reference_png"] = encoded
            response = client.post("/generate", json=request)
            assert response.status_code == 200, response.text
            actual = np.array(Image.open(io.BytesIO(base64.b64decode(response.json()["png_base64"]))))
            expected = runtime.generate(["red bottle"], [image] if image is not None else None, seeds=[7], steps=2)[0]
            assert np.array_equal(actual, expected)
        assert client.post("/generate", json={"prompt": "x", "solver": "invalid"}).status_code == 422
        assert client.post("/generate", json={}).status_code == 422


def test_tile_loss_does_not_touch_items():
    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    image, noise = torch.randn_like(pred), torch.randn_like(pred)
    t = torch.full((2,), 0.1)
    noisy = image * 0.9 + noise * 0.1
    mask = torch.zeros(2, dtype=torch.bool)
    assert torch.equal(flow_loss(pred, image, noisy, t, noise, mask, 100),
                       flow_loss(pred, image, noisy, t, noise, mask, 0))


def test_text_only_checkpoint_rejects_reference(tmp_path):
    catalog = fixture(tmp_path / "fixture")
    cfg = tiny_config()
    cfg.task = "text"
    cfg.train.task_mix = {"text": .9, "reference": 0., "joint": 0., "unconditional": .1}
    checkpoint = Trainer(cfg, catalog, tmp_path / "train").run(stop_after=1)
    runtime = Runtime(checkpoint)
    with pytest.raises(ValueError, match="text-only"):
        runtime.generate(["bottle"], [np.zeros((16, 16, 4), np.uint8)], steps=1)


def test_local_hf_encoder_training_and_artifact_identity(tmp_path):
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tower = tmp_path / "tower"
    model = GPT2Model(GPT2Config(n_embd=16, n_layer=2, n_head=2, vocab_size=5, n_positions=64))
    model.save_pretrained(tower)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "green": 2, "bottle": 3, "[EOS]": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]").save_pretrained(tower)
    data = tiny_config().to_dict()
    data["text"] = {"backend": "hf", "model": str(tower), "layers": [1, 2]}
    cfg = Experiment.from_dict(data)
    catalog = fixture(tmp_path / "fixture")
    checkpoint = Trainer(cfg, catalog, tmp_path / "train").run(stop_after=1)
    runtime = Runtime(checkpoint)
    images = runtime.generate(["green bottle"], seeds=[0], steps=1)
    assert images[0].shape == (8, 8, 4)
    assert not any(p.requires_grad for p in runtime.encoder.model.parameters())
    config = tower / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        Runtime(checkpoint)
