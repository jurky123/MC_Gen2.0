import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from PIL import Image

from config import ModelConfig, TrainConfig
from train.train import Trainer
from infer.sample import load_model_from_checkpoint, sample_textures, save_result
from eval.seam import make_tiled_preview, seam_score


def make_synthetic(count=320, size=32, seed=0):
    rng = np.random.RandomState(seed)
    out = ROOT / "data" / "build" / "smoke" / "staging"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        img = np.zeros((size, size, 3), dtype=np.float32)
        kind = rng.randint(4)
        c1 = rng.rand(3)
        c2 = rng.rand(3)
        if kind == 0:
            img[:] = c1
            img[:, ::4] = c2
        elif kind == 1:
            img[:] = c1
            img[::4, :] = c2
        elif kind == 2:
            gx, gy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
            img = (c1[None, None, :] * gy[..., None] + c2[None, None, :] * gx[..., None])
        else:
            img[:] = c1
            for _ in range(8):
                x0, y0 = rng.randint(0, size), rng.randint(0, size)
                img[max(0, y0 - 2) : y0 + 2, max(0, x0 - 2) : x0 + 2] = c2
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        p = out / f"{i:05d}.png"
        Image.fromarray(img).save(p)
        records.append({"path": str(p), "project_id": f"syn{i % 8}", "weak_prompt": ""})
    return records


def write_configs():
    from data.build_mmap import build_mmap

    build_dir = ROOT / "data" / "build" / "smoke"
    build_dir.mkdir(parents=True, exist_ok=True)
    records = make_synthetic()
    build_mmap(build_dir, records, image_size=32, split_by="project_id", seed=0)

    data_yaml = build_dir / "data.yaml"
    data_cfg = {
        "dataset": {
            "images": str(build_dir / "images.uint8.mmap"),
            "metadata": str(build_dir / "metadata.parquet"),
            "splits": str(build_dir / "splits.json"),
            "image_size": 32,
            "channels": 4,
            "toroidal": True,
            "normalize": True,
            "text_mmap": "",
            "text_dim": 768,
            "max_text_tokens": 64,
            "caption_col": "weak_prompt",
        }
    }
    data_yaml.write_text(yaml.safe_dump(data_cfg))

    train_yaml = build_dir / "train.yaml"
    train_cfg = {
        "train": {
            "precision": "bf16",
            "compile": False,
            "seed": 0,
            "steps": 30,
            "log_every": 5,
            "save_every": 15,
            "val_every": 1000,
            "output_dir": str(ROOT / "checkpoints" / "smoke"),
            "optimizer": {"name": "adamw", "fused": False, "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.03},
            "scheduler": {"type": "cosine", "warmup_steps": 5},
            "grad_clip": 1.0,
            "activation_checkpointing": False,
            "gradient_accumulation": 1,
            "batch": {"auto_probe": False, "preferred_micro_batch": 8, "max_vram_gb": 7.2},
            "ema": {"enabled": True, "device": "cpu", "decay": 0.999, "update_every": 1},
            "flow": {"timestep_sampling": "uniform", "loss": "mse"},
            "tile_loss": {"enabled": True, "weight": 0.03, "max_t": 0.7, "border_width": 2},
            "dataset": str(data_yaml),
            "text_mmap": "",
        }
    }
    train_yaml.write_text(yaml.safe_dump(train_cfg))
    return data_yaml, train_yaml


def main():
    data_yaml, train_yaml = write_configs()
    mcfg = ModelConfig.from_yaml(ROOT / "configs" / "model" / "tiny.yaml")
    tcfg = TrainConfig.from_yaml(train_yaml)
    print(f"params={MCFlowDiT_count(mcfg)}")
    trainer = Trainer(mcfg, tcfg)
    trainer.train(str(data_yaml))

    ckpt = ROOT / "checkpoints" / "smoke" / "latest.pt"
    model, mcfg2, _ = load_model_from_checkpoint(ckpt, "cpu")
    prompts = ["mossy stone bricks", "oak planks with cracks"]
    imgs = sample_textures(model, prompts, seeds=[1, 2], steps=8, cfg=2.0, device="cpu", text_dim=mcfg2.text_dim, max_tokens=mcfg2.max_text_tokens)
    out = ROOT / "outputs" / "smoke"
    for i, (img, p) in enumerate(zip(imgs, prompts)):
        save_result(img, out / f"{i:02d}.png")
        arr = img.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(make_tiled_preview(arr)).save(out / f"{i:02d}.tiled.png")
        print(f"sample {i} seam={seam_score(arr)[0]:.3f} prompt='{p}'")
    print("SMOKE TEST OK")


def MCFlowDiT_count(mcfg):
    from model.mc_flow_dit import MCFlowDiT

    return MCFlowDiT(mcfg).param_count()


if __name__ == "__main__":
    main()
