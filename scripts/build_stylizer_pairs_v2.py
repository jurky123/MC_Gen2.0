"""Build the Stylizer pair dataset from the MC->HD production run.

    ref64  = HD (from the FLUX edit run) resized to 64x64
    target = the REAL MC texture (32x32 RGBA) that the HD was derived from
    text   = the fine prompt used to generate the HD

This is the corrected supervision: the target is the real MC distribution (not
a deterministic downsample of the reference), so the Stylizer learns
"HD -> real MC style", with the HD supplying structure/detail.

    python scripts/build_stylizer_pairs_v2.py --sources pairs/mchd_stage3_a,pairs/mchd_stage3_b \
        --out pairs/stylizer_v2
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from data.mmap_io import write_raw_mmap, write_schema  # noqa: E402
from data.lineage_split import group_split  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="pairs/mchd_stage3_a,pairs/mchd_stage3_b")
    ap.add_argument("--out", default=str(ROOT / "pairs/stylizer_v2"))
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--target-size", type=int, default=32)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build = Path(BUILD)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))

    records = []
    for src in args.sources.split(","):
        src = Path(src)
        mpath = src / "manifest.jsonl"
        if not mpath.exists():
            continue
        for line in mpath.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            hp = Path(r["hd_png"])
            hd_abs = hp if hp.is_absolute() else src / hp
            if hd_abs.exists():
                r["_hd_abs"] = str(hd_abs)
                records.append(r)
    if not records:
        raise SystemExit("no records found")
    # de-dup by row (a shard may have been re-run)
    seen, uniq = set(), []
    for r in records:
        if r["row"] in seen:
            continue
        seen.add(r["row"])
        uniq.append(r)
    records = sorted(uniq, key=lambda r: r["row"])
    n = len(records)
    print(f"{n} pairs from {args.sources}")

    rs, ts = args.ref_size, args.target_size
    ref = np.zeros((n, rs, rs, 4), np.uint8)
    tgt = np.zeros((n, ts, ts, 4), np.uint8)
    rows = []
    for i, r in enumerate(records):
        hd = Image.open(r["_hd_abs"]).convert("RGB")
        ref[i] = np.dstack([np.asarray(hd.resize((rs, rs), Image.Resampling.LANCZOS)),
                            np.full((rs, rs), 255, np.uint8)])
        tgt[i] = np.asarray(imgs[r["row"]])
        rows.append({
            "index": i, "row": int(r["row"]), "prompt": str(r["prompt"]),
            "subject": str(r.get("subject", "")),
            "asset_type": str(r.get("asset_type", "block")),
            "project_id": str(meta.at[r["row"], "project_id"]),
            "tileable": str(meta.at[r["row"], "tileable"]) if "tileable" in meta.columns else "unknown",
        })
    df = pd.DataFrame(rows)
    write_raw_mmap(ref, out / "ref.uint8.mmap")
    write_raw_mmap(tgt, out / "target.uint8.mmap")
    write_schema(out, "ref.uint8.mmap", ref.shape, np.uint8,
                 extra={"ref_size": rs, "target_size": ts, "n": n,
                        "target": "real MC"})
    write_schema(out, "target.uint8.mmap", tgt.shape, np.uint8)
    df.to_parquet(out / "metadata.parquet", index=False)

    roles, audit = group_split(df.to_dict("records"), group_keys=("project_id",),
                               seed=args.seed, fracs=(1 - args.val_fraction,
                                                      args.val_fraction, 0.0))
    splits = {k: [i for i, r in sorted(roles.items()) if r == k] for k in ("train", "val", "test")}
    (out / "splits.json").write_text(json.dumps(splits))
    print(f"types: {dict(Counter(r['asset_type'] for r in rows))}")
    print(f"tileable: {dict(Counter(r['tileable'] for r in rows))}")
    print(f"splits: { {k: len(v) for k, v in splits.items()} } (group={audit['n_groups']} projects)")


if __name__ == "__main__":
    main()
