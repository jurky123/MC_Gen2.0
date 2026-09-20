"""Export a diverse 200-asset MC sample (original resolution) for review.

Selection maximises coverage of type / form / material / colour / state,
prefers textured non-trivial images, de-duplicates exactly, and exports:

    <out>/images/<name>.png      original-resolution PNG
    <out>/manifest.csv           index, file_name, type, prompt, labels, project, license
    <out>/contact_sheet.png      preview grid
    <out>.zip                    everything, ready to download

    python scripts/export_mc_sample.py --n 200 --out /tmp/opencode/mc_sample200
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from data.filename_prompts import clean_tokens  # noqa: E402
from scripts.select_stage3_subset import COLORS, FORMS, MATERIALS, STATES  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"
RAW = ROOT / "data/raw/mc_text2image/data"


def labels(name):
    toks = clean_tokens(str(name), keep_parts=True, keep_anim=True, keep_generic=False)
    ts = set(toks)
    form = next((w for w in toks if w in FORMS), "")
    mat = next((w for w in toks if w in MATERIALS), "")
    col = next((w for w in toks if w in COLORS), "")
    state = next((w for w in toks if w in STATES), "")
    return form, mat, col, state, ts


def is_interesting(arr):
    """Reject blank / near-solid / almost fully transparent textures."""
    a = arr[..., 3]
    if (a > 8).mean() < 0.05:
        return False
    rgb = arr[..., :3][a > 100]
    if len(rgb) < 4:
        return False
    if len(np.unique(rgb, axis=0)) < 2:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="/tmp/opencode/mc_sample200")
    args = ap.parse_args()
    n = args.n
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(BUILD / "metadata.parquet")
    gp = pd.read_parquet(BUILD / "grounded_prompts.parquet", columns=["prompt_0"])
    imgs = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))

    # ---- build candidate pool with labels + quality + dedupe --------
    pool = []
    sha_seen = set()
    for i in range(len(meta)):
        fn = str(meta.at[i, "file_name"])
        arr = np.asarray(imgs[i])
        if not is_interesting(arr):
            continue
        h = hashlib.sha1(arr.tobytes()).hexdigest()
        if h in sha_seen:
            continue
        sha_seen.add(h)
        form, mat, col, state, _ = labels(fn)
        pool.append({
            "index": i, "file_name": fn, "type": str(meta.at[i, "type"]),
            "form": form, "material": mat, "colour": col, "state": state,
            "project_id": str(meta.at[i, "project_id"]),
            "license": str(meta.at[i, "license"]),
            "prompt": str(gp["prompt_0"].iloc[i]),
            "shape": arr.shape,
        })

    # ---- stratified pick: (type,form) first, then (type,material/colour/state)
    picked, used = [], set()
    by_ff = defaultdict(list)
    for c in pool:
        by_ff[(c["type"], c["form"] or "other")].append(c)
    keys = sorted(by_ff, key=lambda k: (k[0], k[1]))
    # round 1: 2 per (type, form)
    for k in keys:
        for c in by_ff[k][:2]:
            if len(picked) < n:
                picked.append(c)
                used.add(c["index"])
    # round 2: coverage of (type, material), (type, colour), (type, state)
    for dim in ("material", "colour", "state"):
        seen = Counter((c["type"], c[dim]) for c in picked)
        cands = sorted(pool, key=lambda c: seen[(c["type"], c[dim])])
        for c in cands:
            if len(picked) >= n:
                break
            if c["index"] in used:
                continue
            picked.append(c)
            used.add(c["index"])
    # round 3: fill remaining randomly-ish in deterministic order
    for c in sorted(pool, key=lambda c: c["index"]):
        if len(picked) >= n:
            break
        if c["index"] not in used:
            picked.append(c)
            used.add(c["index"])

    # ---- extract ORIGINAL resolution bytes from the raw parquet ----
    want = {c["index"]: c for c in picked}
    remaining = set(want)
    got = {}
    offset = 0
    for f in sorted(RAW.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=["image", "file_name"]):
            rows = batch.to_pylist()
            for row in rows:
                if offset in remaining:
                    cell = row.get("image") or {}
                    raw = cell.get("bytes")
                    if raw:
                        got[offset] = (raw, str(row.get("file_name", "")))
                        remaining.discard(offset)
                offset += 1
        if not remaining:
            break

    rows_csv = []
    ok = 0
    for c in picked:
        i = c["index"]
        if i in got:
            raw, fn = got[i]
            im = Image.open(io.BytesIO(raw)).convert("RGBA")
            name = f"{i:07d}_{Path(c['file_name']).stem[:60]}.png"
            im.save(out / "images" / name)
            c["exported"] = name
            c["orig_size"] = f"{im.width}x{im.height}"
            ok += 1
        else:  # fall back to the mmap version
            arr = np.asarray(imgs[i])
            name = f"{i:07d}_{Path(c['file_name']).stem[:60]}_32.png"
            Image.fromarray(arr, "RGBA").save(out / "images" / name)
            c["exported"] = name
            c["orig_size"] = "32x32(mmap)"
        rows_csv.append([c["index"], c["exported"], c["file_name"], c["type"],
                         c["form"], c["material"], c["colour"], c["state"],
                         c["prompt"], c["project_id"], c["license"]])
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "exported_png", "file_name", "type", "form",
                    "material", "colour", "state", "prompt", "project_id", "license"])
        w.writerows(rows_csv)

    # ---- contact sheet ----
    cols = 10
    S = 96
    rowsn = (len(picked) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * S + (cols + 1) * 4, rowsn * S + (rowsn + 1) * 4), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for k, c in enumerate(picked):
        im = Image.open(out / "images" / c["exported"]).convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGB"), mask=im.split()[3])
        x = 4 + (k % cols) * (S + 4)
        y = 4 + (k // cols) * (S + 4)
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (x, y))
    sheet.save(out / "contact_sheet.png")

    # ---- zip ----
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out / "images").glob("*.png")):
            z.write(p, arcname=f"images/{p.name}")
        z.write(out / "manifest.csv", arcname="manifest.csv")
        z.write(out / "contact_sheet.png", arcname="contact_sheet.png")

    dist = {
        "type": dict(Counter(c["type"] for c in picked)),
        "forms": len({c["form"] for c in picked if c["form"]}),
        "materials": len({c["material"] for c in picked if c["material"]}),
        "colours": len({c["colour"] for c in picked if c["colour"]}),
        "states": len({c["state"] for c in picked if c["state"]}),
        "originals_extracted": ok,
    }
    print(json.dumps(dist, indent=1))
    print(f"zip: {zpath}  ({zpath.stat().st_size/1e6:.1f} MB, {len(picked)} images)")


if __name__ == "__main__":
    main()
