import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import load_yaml
from data.build_mmap import build_mmap


def build_materialmaker_stage(build, seed=0):
    rng = np.random.RandomState(seed)
    out_dir = Path(build["out_dir"])
    source = Path(build["materialmaker_dir"])
    target = int(build.get("target_size", 32))
    total = int(build.get("total_target", 100000))
    if not source.exists():
        print("Material Maker source dir not found")
        return
    files = sorted(source.rglob("*.png")) + sorted(source.rglob("*.jpg"))
    if not files:
        print("no procedural texture files found")
        return
    stage_dir = out_dir / "staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(min(total, len(files))):
        img = Image.open(files[i]).convert("RGB")
        out_img = img.resize((target, target), Image.Resampling.BICUBIC)
        dst = stage_dir / f"{i:07d}.png"
        out_img.save(dst)
        records.append({"path": str(dst), "material": files[i].parent.name, "weak_prompt": ""})
    build_mmap(out_dir, records, image_size=target, seed=seed)
    print(f"material maker stage built: {len(records)} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/stage_a.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if "materialmaker_dir" not in cfg["build"]:
        print("config has no materialmaker_dir")
        return
    build_materialmaker_stage(cfg["build"], seed=int(cfg["build"].get("seed", 0)))


if __name__ == "__main__":
    main()