"""Offline real-tower integration check; no long training run or weight updates."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcgen.catalog import atomic_json
from mcgen.conditioning import TextEncoder
from mcgen.config import Experiment
from mcgen.model import PixelFlow


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this two-device integration check requires both available CUDA GPUs")
    cfg = Experiment.load(ROOT / "configs/v3/item_joint.yaml")
    torch.manual_seed(0)
    tower = TextEncoder(cfg.text, cfg.model, "cuda:1")
    prompts = ["red crystal sword item", "mossy stone bricks block"]
    hidden, mask = tower.encode(prompts, "cuda:0")
    assert hidden.shape[-1] == 12288 and torch.isfinite(hidden).all()
    assert hidden.shape[:2] == mask.shape and mask.any(1).all()
    assert not any(p.requires_grad for p in tower.model.parameters())
    model = PixelFlow(cfg.model).to("cuda:0")
    x = torch.randn(2, 4, cfg.model.size, cfg.model.size, device="cuda:0")
    ref = torch.randn(2, 4, cfg.model.ref_size, cfg.model.ref_size, device="cuda:0")
    t = torch.tensor([0.25, 0.75], device="cuda:0")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context = model.context(hidden, mask, ref, torch.tensor([True, False], device="cuda:0"))
        prediction = model(x, t, context)
        loss = (prediction.float() - x).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in tower.model.parameters())
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    report = {"model_path": cfg.text.model, "revision": cfg.text.revision,
              "layers": cfg.text.layers, "hidden_shape": list(hidden.shape),
              "valid_tokens": mask.sum(1).tolist(), "text_identity": tower.identity,
              "tower_frozen": True, "hidden_finite": True, "backward_finite": True,
              "flow_loss": loss.item(), "weight_updates": 0,
              "peak_allocated_gib": {str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in (0, 1)}}
    path = ROOT / "outputs/design_rebuild_20261101/qwen_integration.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
