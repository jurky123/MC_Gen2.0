"""Normalize and exactly deduplicate weak-labelled Minecraft textures."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def run(manifest, out, target=32):
    out = Path(out)
    tiles = out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    seen = set()
    read = written = missing = errors = 0
    with Path(manifest).open(encoding="utf-8") as source, \
            (out / "manifest.jsonl").open("w", encoding="utf-8") as sink:
        for line in source:
            read += 1
            rec = json.loads(line)
            path = Path(rec["path"])
            if not path.is_file():
                missing += 1
                continue
            try:
                with Image.open(path) as image:
                    tile = image.convert("RGB").resize((target, target), Image.Resampling.NEAREST)
            except (OSError, ValueError):
                errors += 1
                continue
            digest = hashlib.sha256(tile.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            dst = tiles / digest[:2] / f"{digest}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tile.save(dst, optimize=True)
            rec.update(path=str(dst), source=str(path),
                       container=str(path.parts[len(Path('data/raw/mc/extracted').parts)])
                       if len(path.parts) > len(Path('data/raw/mc/extracted').parts) else str(path.parent),
                       method="mc_texture", target=target, content_sha256=digest)
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    summary = {"read": read, "written": written, "duplicates": read - written - missing - errors,
               "missing": missing, "errors": errors, "target": target}
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--out", default="data/processed/modrinth32")
    parser.add_argument("--target", type=int, default=32)
    args = parser.parse_args()
    run(args.manifest, args.out, args.target)
