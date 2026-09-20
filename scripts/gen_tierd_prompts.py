"""Tier-D prompt generator: 11-field structured prompts in 3 buckets.

Bucket 1 (vocab stratified, ~60%): cover forms/materials/colours/states.
Bucket 2 (weakness-targeted, ~25%): hand-listed failure modes from evals.
Bucket 3 (novel combos, ~15%): (material, form) pairs absent from real data.

Output JSONL: one object per prompt with all template fields + metadata
(bucket, novelty flag). HD rendering suffixes are appended by
build_tierd_prototype.py.

    python scripts/gen_tierd_prompts.py --n 2000 --out tierd_prompts_2k.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from data.filename_prompts import clean_tokens  # noqa: E402
from scripts.select_stage3_subset import COLORS, FORMS, MATERIALS, STATES  # noqa: E402

FORMS = sorted(FORMS)
MATERIALS = sorted(MATERIALS)
COLORS = sorted(COLORS)
STATES = sorted(STATES)

# Bucket 2: eval failure modes (rare items, composite patterns, emissive,
# transparent, thin structures). Each entry partially specifies fields.
WEAKNESS_SEEDS = [
    {"form": "lyre", "material": "wood", "asset": "item"},
    {"form": "gear", "material": "bronze", "asset": "item"},
    {"form": "key", "material": "copper", "asset": "item"},
    {"form": "crown", "material": "gold", "asset": "item"},
    {"form": "dagger", "material": "steel", "asset": "item"},
    {"form": "ring", "material": "silver", "asset": "item"},
    {"form": "helmet", "material": "steel", "asset": "item"},
    {"form": "potion", "material": "glass", "asset": "item", "state": "glowing"},
    {"form": "bottle", "material": "glass", "asset": "item"},
    {"form": "lantern", "material": "iron", "asset": "block", "state": "glowing"},
    {"form": "lamp", "material": "copper", "asset": "block", "state": "glowing"},
    {"form": "ore", "material": "amethyst", "asset": "block"},
    {"form": "ore", "material": "ruby", "asset": "block"},
    {"form": "ore", "material": "uranium", "asset": "block"},
    {"form": "carpet", "material": "wool", "asset": "block", "pattern": "dot grid"},
    {"form": "tile", "material": "marble", "asset": "block", "pattern": "grid"},
    {"form": "planks", "material": "wood", "asset": "block", "pattern": "stripes"},
    {"form": "bricks", "material": "stone", "asset": "block", "pattern": "brick courses"},
    {"form": "mat", "material": "straw", "asset": "block", "pattern": "braid"},
    {"form": "door", "material": "metal", "asset": "item"},
    {"form": "arrow", "material": "wood", "asset": "item"},
    {"form": "barrel", "material": "wood", "asset": "block"},
    {"form": "chest", "material": "oak", "asset": "block"},
    {"form": "ice", "material": "ice", "asset": "block", "state": "frozen"},
]

PATTERNS = ["brick courses", "dot grid", "grid", "stripes", "braid", "veins",
            "checker", "border frame", "radial", "speckle", "cracks"]
SILHOUETTES = ["square", "round", "tall", "flat", "symmetric", "compact"]
DETAILS = ["rivets", "carved edge", "metal bands", "jewel inlay", "handle",
           "spout", "cap", "hinge", "runes", "studs"]


def present_pairs(meta, limit=400000):
    """(material, form) pairs present in real filenames."""
    present = set()
    files = meta["file_name"].astype(str).tolist()
    step = max(1, len(files) // limit)
    for f in files[::step]:
        toks = set(clean_tokens(f, keep_parts=True, keep_anim=True, keep_generic=False))
        forms = [w for w in toks if w in MATERIALS or w in FORMS]
        mats = [w for w in toks if w in MATERIALS]
        fms = [w for w in toks if w in FORMS]
        for a in mats:
            for b in fms:
                present.add((a, b))
    return present


def render(spec):
    parts = [spec.get("state", ""), spec.get("colour", ""),
             spec.get("material", ""), spec.get("form", "")]
    head = " ".join(p for p in parts if p).strip()
    at = spec["asset"]
    bits = [head + f" {at}"]
    if spec.get("pattern") and spec["pattern"] != "plain":
        bits.append(f"with {spec['pattern']} pattern")
    if spec.get("details"):
        bits.append(f"with {spec['details']}")
    if spec.get("silhouette"):
        bits.append(f"{spec['silhouette']} shape")
    return ", ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--buckets", default="0.6,0.25,0.15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/tierd_prompts_2k.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    b1, b2, b3 = (float(x) for x in args.buckets.split(","))
    n1, n2, n3 = int(args.n * b1), int(args.n * b2), args.n - int(args.n * b1) - int(args.n * b2)

    meta = pd.read_parquet(Path(args.build) / "metadata.parquet",
                           columns=["file_name", "type"])
    present = present_pairs(meta)

    out = []
    # bucket 1: stratified over (asset, form, material)
    for i in range(n1):
        asset = "block" if i % 2 == 0 else "item"
        out.append({
            "asset_type": asset,
            "material": rng.choice(MATERIALS),
            "form": rng.choice(FORMS),
            "dominant_colors": rng.choice(COLORS),
            "silhouette": rng.choice(SILHOUETTES),
            "surface_pattern": rng.choice(PATTERNS + ["plain"]),
            "details": rng.choice(DETAILS + [""]),
            "symmetry": rng.choice(["symmetric", "asymmetric", ""]),
            "emissive": "", "transparency": "",
            "tileability": "tileable" if asset == "block" else "single sprite",
            "orientation": "front view",
            "bucket": 1, "novelty": 0,
        })
    # bucket 2: weakness seeds + random fill
    for i in range(n2):
        seed = dict(rng.choice(WEAKNESS_SEEDS))
        out.append({
            "asset_type": seed.get("asset", "block"),
            "material": seed.get("material", rng.choice(MATERIALS)),
            "form": seed.get("form", rng.choice(FORMS)),
            "dominant_colors": rng.choice(COLORS),
            "silhouette": rng.choice(SILHOUETTES),
            "surface_pattern": seed.get("pattern", rng.choice(PATTERNS + ["plain"])),
            "details": rng.choice(DETAILS + [""]),
            "symmetry": rng.choice(["symmetric", "asymmetric", ""]),
            "emissive": "glowing" if seed.get("state") == "glowing" else "",
            "transparency": "transparent" if seed.get("material") == "glass" else "",
            "tileability": "tileable" if seed.get("asset", "block") == "block" else "single sprite",
            "orientation": "front view",
            "bucket": 2, "novelty": 0,
        })
    # bucket 3: novel (material, form) combos
    made = 0
    guard = 0
    while made < n3 and guard < n3 * 50:
        guard += 1
        asset = rng.choice(["block", "item"])
        mat, frm = rng.choice(MATERIALS), rng.choice(FORMS)
        if (mat, frm) in present:
            continue
        out.append({
            "asset_type": asset,
            "material": mat, "form": frm,
            "dominant_colors": rng.choice(COLORS),
            "silhouette": rng.choice(SILHOUETTES),
            "surface_pattern": rng.choice(PATTERNS + ["plain"]),
            "details": rng.choice(DETAILS + [""]),
            "symmetry": rng.choice(["symmetric", "asymmetric", ""]),
            "emissive": "", "transparency": "",
            "tileability": "tileable" if asset == "block" else "single sprite",
            "orientation": "front view",
            "bucket": 3, "novelty": 1,
        })
        made += 1
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for spec in out:
            spec["prompt"] = render({
                "state": "", "colour": spec["dominant_colors"],
                "material": spec["material"], "form": spec["form"],
                "asset": spec["asset_type"], "pattern": spec["surface_pattern"],
                "details": spec["details"], "silhouette": spec["silhouette"]})
            f.write(json.dumps(spec) + "\n")
    cnt = Counter((s["bucket"], s["asset_type"]) for s in out)
    print(f"wrote {len(out)} prompts -> {args.out}")
    print(dict(cnt))


if __name__ == "__main__":
    main()
