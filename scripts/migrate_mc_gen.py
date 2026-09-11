import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from data.build_mmap import build_mmap

SRC = Path(r"E:\SomeDocs\MC_Gen\data\dataset")


def _to_rgb(src_img, dst, composite="white"):
    im = Image.open(src_img)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, composite)
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    im.save(dst)
    return im


def build_weak_prompt(label, category):
    name = (label or "").lower().strip()
    kind = "block texture" if category == "block" else ("item texture" if category == "item" else "texture")
    return f"{name}, pixel-art minecraft {kind}"


def migrate(out_dir, build_dir, size=16, categories=None, max_records=None, copy_images=True, composite="white"):
    categories = categories or ["block", "item", "fluid"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(l) for l in open(SRC / "labels.jsonl", encoding="utf-8") if l.strip()]
    records = []
    for r in labels:
        if r.get("category") not in categories:
            continue
        if r.get("width") != r.get("height"):
            continue
        src_img = SRC / "images" / r["filename"]
        if not src_img.exists():
            continue
        dst = img_dir / r["filename"]
        if copy_images:
            if not dst.exists():
                _to_rgb(src_img, dst, composite=composite)
        else:
            dst = src_img
        rec = {
            "path": str(dst),
            "project_id": r["mod_id"],
            "mod_id": r["mod_id"],
            "category": r["category"],
            "raw_name": r["raw_name"],
            "label": r["label"],
            "jar": r.get("jar", ""),
            "weak_prompt": build_weak_prompt(r["label"], r["category"]),
        }
        records.append(rec)
        if max_records and len(records) >= max_records:
            break

    build_dir = Path(build_dir)
    build_mmap(build_dir, records, image_size=size, split_by="project_id")
    with open(build_dir / "weak_labels.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"migrated {len(records)} records -> {build_dir} (image_size={size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/mc_gen")
    ap.add_argument("--build-dir", default="data/build/mc_b16")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--categories", nargs="*", default=["block", "item", "fluid"])
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--composite", default="white")
    args = ap.parse_args()
    migrate(
        args.out,
        args.build_dir,
        size=args.size,
        categories=args.categories,
        max_records=args.max or None,
        composite=args.composite,
    )


if __name__ == "__main__":
    main()