"""Build the Stage 2 (MC weak-label) dataset.

Stage 2 = all MC textures with weak text labels + (optionally) curated MC tiles,
used to align the pixel-space DiT to text before the Qwen embedding pipeline.

`data/build/mc_text2image32` has a single placeholder `weak_prompt`, so derive a
usable label from `file_name` (underscores / dashes -> spaces) and `type`.
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data.pixel_training_builder import build


def derive_prompt(row):
    name = str(row.get("file_name") or "").rsplit(".", 1)[0]
    name = re.sub(r"[_\-]+", " ", name).strip()
    typ = str(row.get("type") or "texture").strip() or "texture"
    if not name:
        name = "minecraft"
    return f"{name}, minecraft pixel art {typ} texture"


def prepare_mc(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("images.uint8.mmap", "splits.json", "build_summary.json"):
        s = src / f
        if not s.exists():
            continue
        d = dst / f
        if d.exists():
            d.unlink()
        os.link(s, d)
    df = pd.read_parquet(src / "metadata.parquet")
    df["weak_prompt"] = df.apply(derive_prompt, axis=1)
    df.to_parquet(dst / "metadata.parquet", index=False)
    print(f"prepared {dst}: {len(df)} rows")
    for p in df["weak_prompt"].head(5).tolist():
        print("  ", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", default="data/build/mc_text2image32")
    ap.add_argument("--mc-out", default="data/build/mc_text2image32_wl")
    ap.add_argument("--manifests", nargs="*",
                    default=["data/processed/modrinth32/manifest.jsonl"])
    ap.add_argument("--out", default="data/build/stage2_32")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    prepare_mc(args.mc, args.mc_out)
    if args.prepare_only:
        return
    build(args.manifests, [args.mc_out], args.out, size=32, chunk_size=8192, channels=4)


if __name__ == "__main__":
    main()
