"""Select a category-balanced Stage-3 subset using the label vocabulary.

Vocabulary (counted over all source labels):
  adjectives: 30 colours + 48 states
  nouns:      63 forms + 59 materials
plus orientation/part words (top/side/blade/handle/...).

Rows are classified by these words, then multi-round stratified sampling adds
rows until the target size so that forms, materials, colours, states and their
combinations are all represented.

    python scripts/select_stage3_subset.py --size 20000
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.filename_prompts import MOD_NOISE, label_tokens, type_word  # noqa: E402

FORMS = set("""sword pickaxe axe shovel hoe bow arrow helmet chestplate leggings
boots ingot nugget gem crystal potion bottle door trapdoor ladder slab stairs wall
pane glass leaves log planks plank bricks brick ore wool carpet sapling seed food
dust rod stick fence gate sign button lever torch lantern lamp candle banner bed
chest barrel table plate gear pipe circuit panel tile cog screw wire""".split())

MATERIALS = set("""stone cobblestone deepslate granite diorite andesite basalt
obsidian sandstone terracotta concrete clay wood oak birch spruce jungle acacia
mangrove cherry bamboo crimson warped iron gold copper diamond emerald netherite
amethyst quartz coal redstone lapis ice snow dirt sand gravel mud grass moss bone
leather fabric paper metal steel tin nickel lead zinc uranium aluminum slime
honey resin porcelain ceramic rubber plastic""".split())

COLORS = set("""red orange yellow green cyan blue purple magenta pink brown black
white gray grey silver golden teal lime olive maroon navy beige tan violet indigo
turquoise crimson amber bronze copper""".split())

STATES = set("""mossy cracked polished weathered dark light smooth rough raw
refined broken tall small large old new frozen molten shiny dirty painted aged
ancient rusty damaged glowing burning charged reinforced carved chiseled cut
stripped sealed hidden bright pale deep dull matte glossy wet dry empty full open
closed active inactive""".split())


def classify(tokens):
    words = set(tokens)
    form = next((w for w in tokens if w in FORMS), "other")
    material = next((w for w in tokens if w in MATERIALS), "other")
    colour = next((w for w in tokens if w in COLORS), "other")
    state = next((w for w in tokens if w in STATES), "other")
    return form, material, colour, state


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--max-per-project", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build = Path(args.build)
    meta = pd.read_parquet(build / "metadata.parquet",
                           columns=["type", "file_name", "mod_slug", "project_id"])
    rng = random.Random(args.seed)

    records = []
    for i, row in enumerate(meta.to_dict(orient="records")):
        tokens = label_tokens(row, MOD_NOISE)
        form, material, colour, state = classify(tokens)
        records.append({"index": i, "type": type_word(row), "tokens": tokens,
                        "form": form, "material": material, "colour": colour,
                        "state": state,
                        "project": row.get("project_id") or row.get("mod_slug") or "",
                        "file_name": row.get("file_name")})

    picked = {}

    # rounds of stratified sampling: (keys, cap)
    rounds = [
        (("type", "form"), 40),
        (("type", "material"), 30),
        (("type", "colour"), 30),
        (("type", "state"), 20),
        (("type", "form", "colour"), 6),
        (("type", "material", "state"), 6),
        (("type", "form", "material"), 6),
    ]
    for keys, cap in rounds:
        cells = defaultdict(list)
        for r in records:
            if r["index"] in picked:
                continue
            cells[tuple(r[k] for k in keys)].append(r)
        for items in cells.values():
            rng.shuffle(items)
            for r in items[:cap]:
                picked[r["index"]] = r

    # fill randomly to the target size
    if len(picked) < args.size:
        rest = [r for r in records if r["index"] not in picked]
        rng.shuffle(rest)
        for r in rest:
            if len(picked) >= args.size:
                break
            picked[r["index"]] = r

    # cap per project so no single mod dominates
    by_proj = defaultdict(list)
    for r in picked.values():
        by_proj[r["project"]].append(r)
    final = []
    for items in by_proj.values():
        rng.shuffle(items)
        final.extend(items[:args.max_per_project])
    rng.shuffle(final)
    final = sorted(final[:args.size], key=lambda r: r["index"])

    out = Path(args.out) if args.out else build / "stage3_subset.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for r in final:
            handle.write(json.dumps({k: r[k] for k in
                                     ("index", "type", "form", "material", "colour",
                                      "state", "project", "file_name")},
                                    ensure_ascii=False) + "\n")

    def cover(key, vocab):
        got = {r[key] for r in final} - {"other"}
        return f"{len(got)}/{len(vocab)}"
    t = Counter(r["type"] for r in final)
    print(f"selected {len(final)} / {len(records)} -> {out}")
    print("type:", dict(t))
    print("form coverage:", cover("form", FORMS),
          "| material:", cover("material", MATERIALS),
          "| colour:", cover("colour", COLORS),
          "| state:", cover("state", STATES))
    print("distinct projects:", len(by_proj))
    for key in ("form", "material", "colour", "state"):
        top = Counter(r[key] for r in final).most_common()
        print(f"  {key}: {[f'{w}:{c}' for w, c in top[:12]]}")
    print("  other(no form):", sum(1 for r in final if r["form"] == "other"),
          "| other(no material):", sum(1 for r in final if r["material"] == "other"))


if __name__ == "__main__":
    main()
