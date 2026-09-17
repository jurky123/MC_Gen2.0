"""Build prompt text for every row from source filenames / metadata.

Writes ``filename_prompts.parquet`` aligned to the dataset row order with
``prompt_0..prompt_{K-1}`` columns, so training can read prompts as strings
(dynamic encoding) via ``prompt_cols``.

    python scripts/build_filename_prompts.py \
        --build data/build/mc_text2image32_wl --views 3
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.filename_prompts import build_prompts, clean_tokens  # noqa: E402

# common mod ids that leak into filenames but are not object concepts
MOD_NOISE = {
    "tconstruct", "tinkers", "tcon", "thermal", "mekanism", "create", "botania",
    "ae2", "appliedenergistics", "immersiveengineering", "gtceu", "gregtech",
    "curios", "patchouli", "jei", "top", "jade", "ftb", "forge", "fabric",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--views", type=int, default=3)
    args = ap.parse_args()

    build_dir = Path(args.build)
    meta = pd.read_parquet(build_dir / "metadata.parquet")
    records = meta.to_dict(orient="records")
    n = len(records)
    print(f"rows={n}")

    cols = [[] for _ in range(args.views)]
    empty = 0
    for record in records:
        prompts = build_prompts(record, extra_noise=MOD_NOISE)
        if not prompts:
            empty += 1
            prompts = ["unknown"]
        for i in range(args.views):
            cols[i].append(prompts[i] if i < len(prompts) else prompts[-1])

    out = {"index": meta["index"].values if "index" in meta.columns else list(range(n))}
    for i in range(args.views):
        out[f"prompt_{i}"] = cols[i]
        out[f"style_{i}"] = ["concept", "phrase", "structured"][i] if i < 3 else f"view{i}"
    df = pd.DataFrame(out)
    out_path = Path(args.out) if args.out else build_dir / "filename_prompts.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows={len(df)} empty_concepts={empty}")

    # coverage report (word boundary) on the concept column
    concept = df["prompt_0"].astype(str).str.lower()
    print("\n== concept coverage (word-boundary, full set) ==")
    for w in ["sword", "blade", "lace", "lava", "ember", "potion", "bottle",
              "ingot", "cobblestone", "stone", "brick", "plank", "wood", "iron",
              "metal", "leaf", "leaves", "grass", "heart", "emblem", "crystal",
              "ore", "sand", "glass", "wool", "terracotta", "copper", "diamond"]:
        c = concept.str.contains(rf"\b{w}\b", regex=True).sum()
        print(f"  {w:12s} {c:8d}  ({100*c/len(df):.3f}%)")

    print("\n== examples ==")
    for _, row in df.sample(15, random_state=7).iterrows():
        print("  ", row["prompt_0"], "||", row["prompt_1"], "||", row["prompt_2"])


if __name__ == "__main__":
    main()
