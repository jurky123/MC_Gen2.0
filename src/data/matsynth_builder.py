import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import load_yaml
from data.build_mmap import _crop_resize, build_mmap

BASE_COLOR_NAMES = ("basecolor", "albedo", "color", "diffuse", "col")


def find_basecolor(folder):
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        n = p.stem.lower()
        if any(k in n for k in BASE_COLOR_NAMES):
            return p
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return p
    return None


def scan_matsynth(mat_dir):
    root = Path(mat_dir)
    if not root.exists():
        return []
    mats = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        img = find_basecolor(sub)
        if img is not None:
            mats.append({"material": sub.name, "path": img})
    return mats


def build_generic_stage(build, seed=0):
    rng = np.random.RandomState(seed)
    out_dir = Path(build["out_dir"])
    target = int(build.get("target_size", 32))
    crop_sizes = [int(c) for c in build.get("crop_sizes", [128, 256, 512, 1024])]
    per_range = tuple(build.get("patches_per_material", [32, 128]))
    total = int(build.get("total_target", 250000))

    mats = scan_matsynth(build["matsynth_dir"])
    if not mats:
        print("no MatSynth basecolor images found")
        return
    stage_dir = out_dir / "staging"
    stage_dir.mkdir(parents=True, exist_ok=True)

    records = []
    i = 0
    while i < total:
        m = mats[rng.randint(len(mats))]
        n_patches = rng.randint(*per_range)
        img = Image.open(m["path"]).convert("RGB")
        for _ in range(n_patches):
            if i >= total:
                break
            crop = rng.choice(crop_sizes)
            out_img = _crop_resize(img, crop, target, rng)
            dst = stage_dir / f"{i:07d}.png"
            out_img.save(dst)
            records.append({"path": str(dst), "material": m["material"], "weak_prompt": ""})
            i += 1
    build_mmap(out_dir, records, image_size=target, seed=seed)
    print(f"generic stage built: {len(records)} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_a.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    build_generic_stage(cfg["build"], seed=int(cfg["build"].get("seed", 0)))


if __name__ == "__main__":
    main()