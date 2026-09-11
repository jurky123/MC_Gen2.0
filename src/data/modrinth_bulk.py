import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.modrinth_crawler import crawl_all
from data.mc_extract import extract_package


def extract_all(download_dir, extract_dir, manifest_path):
    download_dir = Path(download_dir)
    manifest = [json.loads(l) for l in open(Path(download_dir) / "crawl_manifest.jsonl", encoding="utf-8") if l.strip()]
    total = 0
    out_manifest = []
    for rec in manifest:
        pid = rec["project_id"]
        z = download_dir / f"{pid}.zip"
        if not z.exists():
            continue
        records = extract_package(z, pid, rec.get("version", ""), rec.get("license_id", ""), extract_dir)
        total += len(records)
        out_manifest.extend(records)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in out_manifest:
            f.write(json.dumps(r) + "\n")
    print(f"extracted {total} textures from {len(manifest)} packages -> {manifest_path}")
    return out_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/mc/downloads")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--types", nargs="*", default=["resourcepack"])
    ap.add_argument("--licenses", nargs="*", default=["CC0-1.0", "CC-BY-4.0"])
    ap.add_argument("--sort", default="downloads")
    ap.add_argument("--categories", nargs="*", default=["16x", "32x", "64x"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--extract-dir", default="data/raw/mc/extracted")
    ap.add_argument("--manifest", default="data/raw/mc/manifest.jsonl")
    args = ap.parse_args()

    if args.dry_run:
        args.extract = False
    crawl_all(args.types, args.licenses, args.out, max_projects=args.max, dry_run=args.dry_run, sort=args.sort, categories=args.categories)
    if args.extract:
        extract_all(args.out, args.extract_dir, args.manifest)


if __name__ == "__main__":
    main()