"""Build the Stylizer v4 pair dataset from BOTH MC->HD generators.

Diversity design: every MC target row contributes up to two pairs that share
the same prompt and the same real-MC target but a different HD reference
(FLUX.2-klein-4B vs Qwen-Image-2.1). The model must map both reference styles
onto the same MC distribution instead of memorising one generator.

    ref    = HD 384 -> white-composite -> LANCZOS -> ref_size (RGBA alpha=255)
    target = the REAL MC texture (32x32 RGBA)
    text   = canonical fine prompt (Stage-3 / grounded, item prefix stripped)

Rows are only taken from the GOOD shards of each generator:
    FLUX : opaque rows from mchd_stage3_*/mchd_fill_* + all mchd_cutout_flux_*
           (the cutouts in the old dirs were white-composited full-frame)
    Qwen : opaque rows (ref_mode==rgb) from mchd_qwen21_* + all mchd_qwen21_cut_*
           (the cutouts in the old dirs had hard stair-step alpha)

    python scripts/build_stylizer_pairs_v4.py --out pairs/stylizer_v4 --ref-size 128
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
from data.ref_preprocess import load_alpha_composited  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"
ITEM_PREFIX = "single item centered on a plain white background, "

FLUX_OLD = ["pairs/mchd_stage3_a", "pairs/mchd_stage3_b",
            "pairs/mchd_fill_a", "pairs/mchd_fill_b"]
FLUX_CUT = ["pairs/mchd_cutout_flux_a", "pairs/mchd_cutout_flux_b"]
QWEN_OLD = ["pairs/mchd_qwen21_a", "pairs/mchd_qwen21_b"]
QWEN_CUT = ["pairs/mchd_qwen21_cut_a", "pairs/mchd_qwen21_cut_b"]


def read_manifest(src):
    p = Path(src) / "manifest.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        hp = Path(r["hd_png"])
        hd = hp if hp.is_absolute() else Path(src) / hp
        if hd.exists():
            r["_hd_abs"] = str(hd)
            out[int(r["row"])] = r
    return out


def canonical_prompt(p):
    p = str(p)
    return p[len(ITEM_PREFIX):] if p.startswith(ITEM_PREFIX) else p


def collect(generator, old_dirs, cut_dirs, keep_old):
    """-> {row: record} using only the good shard of each generator."""
    recs = {}
    cut_rows = set()
    for d in cut_dirs:
        m = read_manifest(d)
        cut_rows |= set(m)
        for row, r in m.items():
            r["_generator"] = generator
            r["_src"] = d
            recs[row] = r
    for d in old_dirs:
        for row, r in read_manifest(d).items():
            if row in cut_rows or row in recs:
                continue  # cutout rows come from the regenerated dirs only
            if not keep_old(row):
                continue
            r["_generator"] = generator
            r["_src"] = d
            recs[row] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "pairs/stylizer_v4"))
    ap.add_argument("--ref-size", type=int, default=128)
    ap.add_argument("--target-size", type=int, default=32)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generators", default="flux,qwen",
                    help="subset of {flux,qwen}")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(BUILD / "metadata.parquet")
    imgs = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3p = pd.read_parquet(BUILD / "stage3_prompts.parquet", columns=["prompt_0"])
    gp = pd.read_parquet(BUILD / "grounded_prompts.parquet", columns=["prompt_0"])
    s3_rows = set()
    sub = BUILD / "stage3_subset.jsonl"
    if sub.exists():
        s3_rows = {int(json.loads(l)["index"]) for l in sub.read_text().splitlines() if l.strip()}

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    sources = {}
    if "flux" in gens:
        # keep opaque rows from the old FLUX dirs (cutouts live in FLUX_CUT)
        sources["flux"] = collect(
            "flux", FLUX_OLD, FLUX_CUT,
            keep_old=lambda row: float(np.mean(np.asarray(imgs[row])[..., 3] == 0)) <= 0.05)
    if "qwen" in gens:
        # Qwen manifests carry ref_mode; opaque == rgb
        q_old_ok = set()
        for d in QWEN_OLD:
            for row, r in read_manifest(d).items():
                if r.get("ref_mode") == "rgb":
                    q_old_ok.add(row)
        sources["qwen"] = collect(
            "qwen", QWEN_OLD, QWEN_CUT, keep_old=lambda row: row in q_old_ok)

    records = []
    for gen, recs in sources.items():
        for row, r in sorted(recs.items()):
            records.append((row, gen, r))
    if not records:
        raise SystemExit("no records found (are the pair dirs built?)")

    # coverage report
    per_gen = Counter(g for _, g, _ in records)
    print(f"pairs: {len(records)} total  {dict(per_gen)}")
    rows_per_gen = {g: {row for row, gg, _ in records if gg == g} for g in per_gen}
    if len(per_gen) > 1:
        common = set.intersection(*rows_per_gen.values())
        print(f"rows with both generators: {len(common)} / "
              f"{len(set.union(*rows_per_gen.values()))}")

    rs, ts = args.ref_size, args.target_size
    n = len(records)
    ref = np.zeros((n, rs, rs, 4), np.uint8)
    tgt = np.zeros((n, ts, ts, 4), np.uint8)
    rows = []
    for i, (row, gen, r) in enumerate(records):
        # same ref construction for both generators: white composite, so a
        # Qwen RGBA cutout and a FLUX RGB cutout look identical to the model.
        rgb = np.asarray(load_alpha_composited(r["_hd_abs"], rs), dtype=np.uint8)
        ref[i] = np.dstack([rgb, np.full((rs, rs), 255, np.uint8)])
        tgt[i] = np.asarray(imgs[row])
        prompt = canonical_prompt(r["prompt"])
        rows.append({
            "index": i, "row": int(row), "prompt": prompt,
            "subject": str(r.get("subject", "")),
            "asset_type": str(r.get("asset_type", "block")),
            "project_id": str(meta.at[row, "project_id"]),
            "tileable": str(meta.at[row, "tileable"]) if "tileable" in meta.columns else "unknown",
            "generator": gen,
            "generator_model": str(r.get("generator_model", "")),
            "generator_license": str(r.get("generator_license", "")),
            "src": str(r["_src"]),
        })
    df = pd.DataFrame(rows)

    write_raw_mmap(ref, out / "ref.uint8.mmap")
    write_raw_mmap(tgt, out / "target.uint8.mmap")
    write_schema(out, "ref.uint8.mmap", ref.shape, np.uint8,
                 extra={"ref_size": rs, "target_size": ts, "n": n,
                        "target": "real MC", "generators": dict(per_gen)})
    write_schema(out, "target.uint8.mmap", tgt.shape, np.uint8)
    df.to_parquet(out / "metadata.parquet", index=False)

    # group by project: both generator variants of a row always land together
    roles, audit = group_split(df.to_dict("records"), group_keys=("project_id",),
                               seed=args.seed, fracs=(1 - args.val_fraction,
                                                      args.val_fraction, 0.0))
    splits = {k: [i for i, r in sorted(roles.items()) if r == k]
              for k in ("train", "val", "test")}
    (out / "splits.json").write_text(json.dumps(splits))
    print(f"types: {dict(Counter(r['asset_type'] for r in rows))}")
    print(f"tileable: {dict(Counter(r['tileable'] for r in rows))}")
    print(f"splits: { {k: len(v) for k, v in splits.items()} } "
          f"(group={audit['n_groups']} projects)")

    # leakage check: no row in both train and val
    row_of = {int(r["index"]): int(r["row"]) for r in rows}
    tr = {row_of[i] for i in splits["train"]}
    va = {row_of[i] for i in splits["val"]}
    if tr & va:
        raise SystemExit(f"row leakage between train/val: {len(tr & va)}")
    print("no train/val row leakage")


if __name__ == "__main__":
    main()
