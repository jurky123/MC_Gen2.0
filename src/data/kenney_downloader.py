import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

UA = "Mozilla/5.0 (mc-texture-gen data acquisition)"

PIXEL_PACKS = [
    "prototype-textures",
    "pattern-pack-pixel",
    "1-bit-pack",
    "block-pack",
    "pixel-platformer",
    "rpg-urban-kit",
    "mini-dungeon",
    "modular-cave-kit",
    "tiny-dungeon",
    "tiny-farm",
    "micro-roguelike",
    "scribble-dungeons",
    "platformer-pack-redux",
    "pixel-shmup",
    "hexagon-pack",
    "bit-pack",
    "roguelike-rpg-pack",
]


def find_zip_url(slug):
    url = f"https://kenney.nl/assets/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "ignore")
    m = re.search(r"https://kenney\.nl/media/pages/assets/[^\"'\s]+\.zip", html)
    return m.group(0) if m else None


def download(url, out):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
        f.write(r.read())
    return out


def run(out_dir, slugs=None, dry_run=False, extract=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slugs = slugs or PIXEL_PACKS
    for slug in slugs:
        try:
            url = find_zip_url(slug)
        except Exception as e:
            print(f"  {slug}: page error {e}")
            continue
        if not url:
            print(f"  {slug}: no zip found (skip)")
            continue
        dst = out_dir / f"{slug}.zip"
        if dst.exists():
            print(f"  {slug}: already downloaded")
            continue
        if dry_run:
            print(f"  {slug}: {url}")
            continue
        try:
            download(url, dst)
            print(f"  {slug}: {dst.stat().st_size // 1000}KB")
            if extract:
                with zipfile.ZipFile(dst) as z:
                    z.extractall(out_dir / slug)
        except Exception as e:
            print(f"  {slug}: download error {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/kenney")
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    run(args.out, slugs=args.slugs, dry_run=args.dry_run, extract=not args.no_extract)


if __name__ == "__main__":
    main()