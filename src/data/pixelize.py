import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import load_yaml
from data.build_mmap import build_mmap


def pixelize(img, low=16, target=32):
    return img.resize((low, low), Image.Resampling.NEAREST).resize((target, target), Image.Resampling.NEAREST)


def quantize(img, n_colors, target=32):
    q = img.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    return q.convert("RGB").resize((target, target), Image.Resampling.NEAREST)


def pixel_pipeline(img, mode, palette_size, target=32):
    if mode == "direct_lowres":
        return img.resize((target, target), Image.Resampling.NEAREST)
    if mode == "palette_quant":
        return quantize(img, palette_size, target)
    if mode == "pixelized_quant":
        return quantize(pixelize(img, target=target), palette_size, target)
    if mode == "raw":
        return img.convert("RGB").resize((target, target), Image.Resampling.NEAREST)
    raise ValueError(mode)


def _sample_choices(rng, fracs, n):
    keys = list(fracs.keys())
    probs = np.array([fracs[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


def build_pixel_stage(build, seed=0):
    rng = np.random.RandomState(seed)
    out_dir = Path(build["out_dir"])
    generic = Path(build["generic_dir"])
    kenney = Path(build["kenney_dir"])
    oga = Path(build["oga_dir"])
    target = int(build.get("target_size", 32))
    total = int(build.get("total_target", 100000))
    mix = build.get("mix", {})
    modes = build.get("pixel_modes", {})
    palettes = build.get("palette_sizes", [8, 16, 32, 64])

    generic_files = sorted(generic.rglob("*.png")) if generic.exists() else []
    kenney_files = sorted(kenney.rglob("*.png")) if kenney.exists() else []
    oga_files = sorted(oga.rglob("*.png")) if oga.exists() else []
    source_map = {"pseudo_pixel_frac": generic_files, "kenney_frac": kenney_files, "oga_frac": oga_files}
    fracs = {k: float(mix.get(k, 0)) for k in source_map}
    n_total = sum(len(v) for v in source_map.values())
    if n_total == 0:
        print("no source images found")
        return
    kinds = _sample_choices(rng, fracs, total)

    records = []
    stage_dir = out_dir / "staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i, kind in enumerate(kinds):
        pool = source_map[kind]
        src = pool[rng.randint(len(pool))]
        img = Image.open(src).convert("RGB")
        mode = _sample_choices(rng, {k: float(v) for k, v in modes.items()}, 1)[0]
        palette = int(rng.choice(palettes))
        out_img = pixel_pipeline(img, mode, palette, target)
        dst = stage_dir / f"{i:07d}.png"
        out_img.save(dst)
        records.append({"path": str(dst), "kind": kind, "mode": mode, "palette": palette, "weak_prompt": ""})
    build_mmap(out_dir, records, image_size=target, split_by="project_id", seed=seed)
    print(f"pixel stage built: {len(records)} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_a5.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    build_pixel_stage(cfg["build"], seed=int(cfg["build"].get("seed", 0)))


if __name__ == "__main__":
    main()