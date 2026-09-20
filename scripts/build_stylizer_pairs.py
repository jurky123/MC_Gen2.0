"""Build the HD->MC Stylizer pair dataset from Tier-D outputs.

For every Tier-D entry:
    ref64   = HD resized to 64x64 (RGBA)
    target  = 90% direct: HD -> (whitekey for items) -> NEAREST 32px  (T2)
              10% sd-edit: the stored SDEdit MC from Tier-D          (T3)

Writes headerless mmaps + prompts/metadata + a random train/val split:
    pairs/stylizer/{ref.uint8.mmap, target.uint8.mmap, prompts.parquet,
                    metadata.parquet, splits.json, schema.json}

    python scripts/build_stylizer_pairs.py --tierd pairs/tierd_17k \
        --out pairs/stylizer --ref-size 64 --target-size 32 --t3-every 10
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from data import mmap_io  # noqa: E402
from data.mmap_io import write_raw_mmap, write_schema  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))


def whitekey_alpha(hd_rgb, thresh=242):
    from scipy.ndimage import binary_propagation

    arr = np.asarray(hd_rgb.convert("RGB"), dtype=np.uint8)
    white = (arr >= thresh).all(axis=-1)
    seeds = np.zeros_like(white)
    seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
    bg = binary_propagation(seeds, mask=white)
    return np.dstack([arr, np.where(bg, 0, 255).astype(np.uint8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tierd", default=str(ROOT / "pairs/tierd_17k"))
    ap.add_argument("--out", default=str(ROOT / "pairs/stylizer"))
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--target-size", type=int, default=32)
    ap.add_argument("--t3-every", type=int, default=10,
                    help="1 in N targets uses the SDEdit variant (T3); rest direct")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tierd = Path(args.tierd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(l) for l in (tierd / "manifest.jsonl").read_text().splitlines()
               if l.strip()]
    n = len(entries)
    rs, ts = args.ref_size, args.target_size
    ref = np.zeros((n, rs, rs, 4), np.uint8)
    tgt = np.zeros((n, ts, ts, 4), np.uint8)
    rows = []
    def resolve(p):
        p = Path(p)
        if p.is_absolute() and p.exists():
            return p
        for cand in (tierd / p, tierd / "hd" / p.name, ROOT / p):
            if cand.exists():
                return cand
        raise FileNotFoundError(p)

    for i, e in enumerate(entries):
        hd = Image.open(resolve(e["hd_png"])).convert("RGB")
        ref[i] = np.dstack([np.asarray(hd.resize((rs, rs), Image.Resampling.LANCZOS)),
                            np.full((rs, rs), 255, np.uint8)])
        role = "t3" if (i % args.t3_every == 0) else "t2"
        if role == "t3":
            t = Image.open(resolve(e["mc_png"])).convert("RGBA")
            tgt[i] = np.asarray(t.resize((ts, ts), Image.Resampling.NEAREST))
        else:
            if e.get("asset_type") == "item":
                arr = whitekey_alpha(hd)
            else:
                arr = np.dstack([np.asarray(hd), np.full((hd.size[1], hd.size[0]), 255, np.uint8)])
            tgt[i] = np.asarray(
                Image.fromarray(arr, "RGBA").resize((ts, ts), Image.Resampling.NEAREST))
        rows.append({
            "index": i, "prompt": str(e.get("prompt", "")),
            "asset_type": e.get("asset_type", "block"),
            "bucket": e.get("bucket", -1), "novelty": e.get("novelty", 0),
            "role": role, "tileable": e.get("asset_type", "block") == "block",
        })
    write_raw_mmap(ref, out / "ref.uint8.mmap")
    write_raw_mmap(tgt, out / "target.uint8.mmap")
    write_schema(out, "ref.uint8.mmap", ref.shape, np.uint8,
                 extra={"ref_size": rs, "target_size": ts, "n": n})
    write_schema(out, "target.uint8.mmap", tgt.shape, np.uint8)
    pd.DataFrame(rows).to_parquet(out / "metadata.parquet", index=False)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    splits = {"train": sorted(int(x) for x in perm[n_val:]),
              "val": sorted(int(x) for x in perm[:n_val])}
    (out / "splits.json").write_text(json.dumps(splits))
    print(f"built {n} pairs -> {out}")
    print("roles:", dict(Counter(r["role"] for r in rows)))
    print("types:", dict(Counter(r["asset_type"] for r in rows)))
    print("splits:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
