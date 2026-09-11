import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image

SUPPORTED_SIZES = {16, 32, 64}
TEXTURE_RE = re.compile(r"^assets/([^/]+)/textures/blocks?/([^/]+\.png)$")


def _iter_zip(path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = TEXTURE_RE.match(name)
            if not m or name.endswith(".png.mcmeta"):
                continue
            ns, rel = m.groups()
            yield ns, rel, z.read(name)


def _iter_dir(path):
    root = Path(path) / "assets"
    if not root.exists():
        root = Path(path)
    for p in root.rglob("*.png"):
        if p.name.endswith(".png.mcmeta"):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        m = TEXTURE_RE.match("assets/" + rel)
        if not m:
            continue
        ns, name = m.groups()
        yield ns, name, p.read_bytes()


def extract_package(pkg_path, project_id, version, license_id, out_dir, supported=SUPPORTED_SIZES):
    out_dir = Path(out_dir)
    records = []
    iter_fn = _iter_dir if Path(pkg_path).is_dir() else _iter_zip
    for ns, rel, data in iter_fn(pkg_path):
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            continue
        if im.size not in [(s, s) for s in supported]:
            continue
        img = Image.new("RGB", im.size)
        img.paste(im.convert("RGB"))
        dst = out_dir / project_id / ns / "block" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        records.append(
            {
                "project_id": project_id,
                "version": version,
                "license_id": license_id,
                "namespace": ns,
                "rel_path": f"{ns}/block/{rel}",
                "size": img.size[0],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--project", default="unknown")
    ap.add_argument("--version", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="data/raw/mc/manifest.jsonl")
    args = ap.parse_args()
    records = extract_package(args.pkg, args.project, args.version, args.license, args.out)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"extracted {len(records)} textures")


if __name__ == "__main__":
    main()