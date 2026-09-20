"""Add a tri-state `tileable` column to metadata.parquet (P1-7).

- Stage-3 subset rows: authoritative 27B annotation field (`yes`/`no`).
- Other rows: explicit keyword allow/deny lists; anything uncertain stays
  `unknown`, which the dataset treats as NOT tileable (conservative: only
  confident tileable samples get the seam/tile loss).
- Row order is preserved and verified; the previous metadata is backed up.

    python scripts/build_tileable_col.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"

TILEABLE_TRUE = (
    "plank", "brick", "ore", "stone", "dirt", "sand", "gravel", "wool",
    "glass", "leaves", "leaff", "log", "concrete", "terracotta", "cobble",
    "moss", "grass", "clay", "snow", "ice", "obsidian", "sandstone",
    "diorite", "granite", "andesite", "basalt", "deepslate", "tuff", "mud",
    "netherrack", "endstone", "purpur", "prismarine", "hay", "melon",
    "sponge", "slime", "honey", "carpet", "mycelium", "podzol", "soulsoil",
    "soulsand", "crying", "glowstone", "sealantern", "shroomlight", "wart",
)
TILEABLE_FALSE = (
    "door", "trapdoor", "sign", "banner", "ladder", "boat", "minecart",
    "rail", "chest", "furnace", "table", "bookshelf", "flowerpot", "pot",
    "bed", "machine", "engine", "button", "lever", "torch", "lantern",
    "lamp", "candle", "gate", "fence", "anvil", "cauldron", "hopper",
    "dropper", "dispenser", "observer", "piston", "jukebox", "brewing",
    "beacon", "conduit", "bell", "flower", "sapling", "seed", "crop",
    "stem", "vine", "mushroom", "coral", "sprout", "roots", "stairs",
    "slab", "pane", "bars", "grate", "hatch", "shutter", "chain", "bell",
    "enchant", "lectern", "loom", "cartography", "stonecutter", "grindstone",
    "smoker", "blast", "composter", "barrel", "shulker", "enderchest",
)


def main():
    build = BUILD
    meta = pd.read_parquet(build / "metadata.parquet")
    n0 = len(meta)
    cols0 = list(meta.columns)

    annot = {}
    for sh in ("shard00", "shard01"):
        p = build / f"stage3_annotations.{sh}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            v = str(d.get("annotation", {}).get("tileable", "")).lower()
            annot[int(d["index"])] = v

    names = meta["file_name"].astype(str).str.lower().to_numpy()
    is_block = (meta["type"].astype(str) == "block").to_numpy()
    is_item = (meta["type"].astype(str) == "item").to_numpy()
    tile = np.full(n0, "unknown", dtype=object)
    matched_true = matched_false = matched_annot = 0
    for i in range(n0):
        if i in annot and annot[i] in ("yes", "no"):
            tile[i] = "true" if annot[i] == "yes" else "false"
            matched_annot += 1
            continue
        if is_item[i]:
            tile[i] = "false"
            continue
        if not is_block[i]:
            continue
        nm = names[i]
        if any(k in nm for k in TILEABLE_FALSE):
            tile[i] = "false"
            matched_false += 1
        elif any(k in nm for k in TILEABLE_TRUE):
            tile[i] = "true"
            matched_true += 1
    meta["tileable"] = tile

    bak = build / "metadata.parquet.bak_no_tileable"
    if not bak.exists():
        shutil.copy2(build / "metadata.parquet", bak)
    meta.to_parquet(build / "metadata.parquet", index=False)
    check = pd.read_parquet(build / "metadata.parquet")
    assert list(check.columns) == cols0 + ["tileable"]
    assert (check[cols0] == meta[cols0]).all().all() or True
    for c in cols0:
        assert check[c].equals(meta[c]), f"column {c} changed!"
    from collections import Counter
    print("tileable:", dict(Counter(tile.tolist())))
    print(f"annot rows: {matched_annot}, keyword true: {matched_true}, false: {matched_false}")


if __name__ == "__main__":
    main()
