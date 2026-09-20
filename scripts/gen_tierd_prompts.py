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

# ---- extended coverage (beyond the stage-3 filename vocab) ----
EXTRA_FORMS = """ring crown key gear lantern shield hammer spear staff wand orb
egg feather flower mushroom shell skull coin chalice book scroll candle anvil
bucket compass clock bell mirror vase statue mask glove boot cloak saddle
sled cart wheel anchor chain rope ladder torch banner flag sail tent bridge
fountain well mill windmill tower dome arch column statue pedestal altar throne
cage trap hook needle pin button badge medal gem necklace bracelet earring
brooch pendant locket flask jar jug mug cup plate bowl pot pan kettle basket
crate sack pouch quiver sheath holster sling pouch vial syringe brush comb
mirror razor hammer tongs shears scissors saw chisel file awl punch stamp seal
die token chip tile domino dice pawn rook knight bishop queen king piece medal
trophy medal ribbon banner pennant kite balloon lantern torch candle lamp
furnace kiln forge bellows crucible mold cast ingot-mold""".split()
EXTRA_MATERIALS = """marble steel bronze brass jade ruby sapphire topaz onyx pearl
coral ivory ebony mahogany straw reed rope chain magma lava glass crystal ice
frost silver-steel darksteel runesteel starmetal moonstone sunstone bloodstone
amber jet coral shell chitin scale feather fur woolen linen silk velvet denim
canvas burlap straw thatch reed bamboo rattan wicker clay brick cobble pebble
gravel chalk limestone sandstone slate granite marble quartz crystal prism
obsidian glass stained-glass mirror chrome copper brass bronze pewter tin lead
nickel zinc aluminum titanium platinum gold rose-gold white-gold electrum""".split()
EXTRA_COLORS = """crimson scarlet azure mint peach lavender charcoal cream rust
moss sage olive teal cyan magenta violet indigo beige tan khaki slate ash snow
ivory pearl coral salmon rose wine burgundy plum eggplant navy royal sky baby
powder mint jade emerald forest pine lime chartreuse olive mustard amber honey
bronze copper rust terracotta brick chocolate coffee caramel toffee sand stone
slate steel iron lead pewter silver chrome platinum gold brass champagne""".split()
ALL_FORMS = sorted(set(FORMS) | set(EXTRA_FORMS))
ALL_MATERIALS = sorted(set(MATERIALS) | set(EXTRA_MATERIALS))
ALL_COLORS = sorted(set(COLORS) | set(EXTRA_COLORS))

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
    {"form": "record", "material": "ebony", "asset": "item"},
    {"form": "helmet", "material": "steel", "asset": "item"},
    {"form": "shield", "material": "oak", "asset": "item"},
    {"form": "staff", "material": "ebony", "asset": "item"},
    {"form": "orb", "material": "crystal", "asset": "item", "state": "glowing"},
    {"form": "coin", "material": "gold", "asset": "item"},
    {"form": "chalice", "material": "silver", "asset": "item"},
    {"form": "book", "material": "leather", "asset": "item"},
    {"form": "scroll", "material": "paper", "asset": "item"},
    {"form": "candle", "material": "wax", "asset": "block", "state": "glowing"},
    {"form": "mushroom", "material": "red", "asset": "block"},
    {"form": "shell", "material": "pearl", "asset": "item"},
    {"form": "skull", "material": "bone", "asset": "block"},
    {"form": "anvil", "material": "iron", "asset": "block"},
    {"form": "compass", "material": "brass", "asset": "item"},
    {"form": "clock", "material": "oak", "asset": "block"},
    {"form": "bell", "material": "bronze", "asset": "item"},
    {"form": "vase", "material": "porcelain", "asset": "block"},
    {"form": "mask", "material": "gold", "asset": "item"},
    {"form": "throne", "material": "oak", "asset": "block"},
    {"form": "fountain", "material": "marble", "asset": "block"},
    {"form": "windmill", "material": "wood", "asset": "block"},
    {"form": "honeycomb", "material": "honey", "asset": "block"},
]

PATTERNS = ["brick courses", "dot grid", "grid", "stripes", "braid", "veins",
            "checker", "border frame", "radial", "speckle", "cracks",
            "horizontal stripes", "vertical stripes", "diagonal stripes",
            "weave", "scales", "ribs", "honeycomb", "spiral", "zigzag",
            "argyle diamonds", "bands", "spots", "stars", "checkerboard",
            "marble veins", "wood grain", "moss patches", "rust patches",
            "gems inset", "rune row", "studs", "rivets", "emblem crest",
            "gradient", "two-tone split", "frame border", "corner ornaments"]
SILHOUETTES = ["square", "round", "tall", "flat", "symmetric", "compact",
               "tiny", "small", "medium", "large", "wide", "narrow",
               "elongated", "stocky", "triangular", "hexagonal", "oval"]
DETAILS = ["rivets", "carved edge", "metal bands", "jewel inlay", "handle",
           "spout", "cap", "hinge", "runes", "studs", "gold trim", "rope wrap",
           "leather grip", "chain links", "gear teeth", "crown points",
           "bottle cork", "key teeth", "helmet visor", "eye slit",
           "chest lock", "drawer pulls", "ladder rungs", "fence posts",
           "arrow fletching", "sword fuller", "shield boss", "gem facets",
           "claw setting", "engraved lines", "embossed edge", "tassel"]


def mine_dataset_prompts(build, n, seed, exclude_rows=()):
    """Bucket 4: sample real prompts previously used in our dataset.

    Sources: grounded_prompts (1M filename-label prompts) + stage3 fine
    prompts (subset rows). Stratified round-robin by project so big packs
    don't dominate; exact-dedup; drop rows used in any holdout.
    """
    import pandas as pd

    rng = random.Random(seed)
    build = Path(build)
    meta = pd.read_parquet(build / "metadata.parquet", columns=["project_id", "type"])
    gp = pd.read_parquet(build / "grounded_prompts.parquet", columns=["prompt_0"])
    subset_idx = {int(json.loads(l)["index"])
                  for l in (build / "stage3_subset.jsonl").read_text().splitlines() if l.strip()}
    s3p = pd.read_parquet(build / "stage3_prompts.parquet", columns=["prompt_0"])
    excluded = set(exclude_rows)
    # candidate pool: (prompt, asset_type, project)
    by_proj = {}
    for i in range(len(meta)):
        if i in excluded:
            continue
        p = str(s3p["prompt_0"].iloc[i]) if i in subset_idx else str(gp["prompt_0"].iloc[i])
        if len(p.split()) < 3:
            continue
        by_proj.setdefault(str(meta["project_id"].iloc[i]),
                           []).append((p, str(meta["type"].iloc[i])))
    projects = sorted(by_proj)
    rng.shuffle(projects)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        for proj in projects:
            if len(out) >= n:
                break
            pool = by_proj[proj]
            p, at = pool[rng.randrange(len(pool))]
            if p in seen:
                continue
            seen.add(p)
            out.append({"asset_type": at if at in ("block", "item") else "block",
                        "prompt": p, "bucket": 4, "novelty": 0,
                        "material": "", "form": "", "dominant_colors": "",
                        "silhouette": "", "surface_pattern": "", "details": "",
                        "symmetry": "", "emissive": "", "transparency": "",
                        "tileability": "", "orientation": "",
                        "source": f"dataset:{proj}"})
    return out


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
    ap.add_argument("--buckets", default="0.4,0.15,0.1,0.35",
                    help="fractions for buckets 1,2,3,4 (4=dataset-mined)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/tierd_prompts_2k.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fracs = [float(x) for x in args.buckets.split(",")]
    assert abs(sum(fracs) - 1.0) < 1e-6 and len(fracs) == 4, "--buckets must be 4 fractions summing to 1"
    b1, b2, b3 = (int(args.n * f) for f in fracs[:3])
    b4 = args.n - b1 - b2 - b3
    n1, n2, n3 = b1, b2, b3

    meta = pd.read_parquet(Path(args.build) / "metadata.parquet",
                           columns=["file_name", "type"])
    present = present_pairs(meta)

    out = []
    # bucket 1: stratified over (asset, form, material)
    for i in range(n1):
        asset = "block" if i % 2 == 0 else "item"
        out.append({
            "asset_type": asset,
            "material": rng.choice(ALL_MATERIALS),
            "form": rng.choice(ALL_FORMS),
            "dominant_colors": rng.choice(ALL_COLORS),
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
            "material": seed.get("material", rng.choice(ALL_MATERIALS)),
            "form": seed.get("form", rng.choice(ALL_FORMS)),
            "dominant_colors": rng.choice(ALL_COLORS),
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
        mat, frm = rng.choice(ALL_MATERIALS), rng.choice(ALL_FORMS)
        if (mat, frm) in present:
            continue
        out.append({
            "asset_type": asset,
            "material": mat, "form": frm,
            "dominant_colors": rng.choice(ALL_COLORS),
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
    # bucket 4: mined real prompts (dedupe against buckets 1-3 as well)
    have = {s.get("prompt", "") for s in out}
    mined = [s for s in mine_dataset_prompts(args.build, b4, args.seed + 7)
             if s["prompt"] not in have]
    have.update(s["prompt"] for s in mined)
    out.extend(mined)
    with open(args.out, "w", encoding="utf-8") as f:
        for spec in out:
            if spec.get("bucket") != 4 or not spec.get("prompt"):
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
