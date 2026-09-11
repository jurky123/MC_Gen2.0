import argparse
import io
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

API = "https://ambientcg.com/api/v3/assets"
UA = "mc-texture-gen/0.1 (data acquisition)"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url, out):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
        f.write(r.read())
    return out


def list_materials(sort="popular", limit=None):
    out = []
    offset = 0
    while True:
        url = f"{API}?type=material&include=downloads&sort={sort}&offset={offset}&limit=100"
        d = _get(url)
        assets = d.get("assets", [])
        if not assets:
            break
        out.extend(assets)
        offset += len(assets)
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def pick_download(asset, resolution):
    for dl in asset.get("downloads", []):
        if dl.get("attributes") == resolution:
            return dl
    return None


def extract_color(zip_path, out_dir, material_id):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            low = base.lower()
            if "_color." not in low:
                continue
            data = z.read(name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            dst = out_dir / material_id / f"{material_id}_Color.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst)
            return dst
    return None


def run(out_dir, limit=None, resolution="1K-JPG", sort="popular", dry_run=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = list_materials(sort=sort, limit=limit)
    print(f"materials: {len(assets)}")
    manifest = []
    ok = skip = fail = 0
    for a in assets:
        mid = a.get("id", "")
        dl = pick_download(a, resolution)
        if dl is None:
            skip += 1
            continue
        color = out_dir / mid / f"{mid}_Color.jpg"
        if color.exists():
            skip += 1
            continue
        url = dl.get("url")
        rec = {"material_id": mid, "url": url, "size": dl.get("size")}
        manifest.append(rec)
        if dry_run:
            continue
        zip_path = out_dir / mid / "raw.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download(url, zip_path)
            dst = extract_color(zip_path, out_dir, mid)
            zip_path.unlink()
            if dst is not None:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  failed {mid}: {e}")
            fail += 1
        if (ok + skip + fail) % 25 == 0:
            print(f"  progress ok={ok} skip={skip} fail={fail}")
    with open(out_dir / "ambientcg_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"done: ok={ok} skip={skip} fail={fail} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/matsynth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resolution", default="1K-JPG")
    ap.add_argument("--sort", default="popular")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.out, limit=args.limit or None, resolution=args.resolution, sort=args.sort, dry_run=args.dry_run)


if __name__ == "__main__":
    main()