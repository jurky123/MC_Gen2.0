"""Split pixel-art sheets into training tiles without extracting archives."""
import argparse, hashlib, io, json, re, sys, zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".tools" / "pixel-processing"))
from scipy import ndimage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
CELL_SIZES = (8, 16, 24, 32, 48, 64)


def hinted_cell(name):
    m = re.search(r"(?<!\d)(8|16|24|32|48|64)\s*[xX]\s*\1(?!\d)", name)
    return int(m.group(1)) if m else None


def choose_cell(img, name, max_tiles=4096):
    w, h = img.size
    if w <= 64 and h <= 64:
        return max(w, h), False
    hint = hinted_cell(name)
    candidates = [c for c in CELL_SIZES if w % c == 0 and h % c == 0 and 2 <= (w // c) * (h // c) <= max_tiles]
    if hint in candidates:
        return hint, True
    # 16/32 are the useful MC-scale defaults; prefer a grid with neither too few nor too many cells.
    for c in (16, 32, 24, 8, 48, 64):
        if c in candidates:
            return c, True
    return None, False


def _runs(values):
    idx = np.flatnonzero(values)
    if not len(idx): return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, cuts + 1]; ends = np.r_[cuts, len(idx) - 1]
    return [(int(idx[a]), int(idx[b]) + 1) for a, b in zip(starts, ends)]


def content_boxes(img, name, max_regions=4096):
    """Use an explicit grid when declared, otherwise find 2-D alpha components."""
    alpha = np.asarray(img.getchannel("A")) > 8
    coverage = float(alpha.mean())
    if img.width <= 64 and img.height <= 64:
        return [(0, 0, img.width, img.height)], "single"
    hint = hinted_cell(name)
    if hint and img.width % hint == 0 and img.height % hint == 0:
        n = (img.width // hint) * (img.height // hint)
        if 2 <= n <= max_regions:
            return [(x, y, x+hint, y+hint) for y in range(0, img.height, hint)
                    for x in range(0, img.width, hint)], "hinted_grid"
    if coverage < 0.98 and alpha.size <= 16_000_000:
        merged = ndimage.binary_dilation(alpha, iterations=1)
        labels, count = ndimage.label(merged)
        if count <= max_regions:
            boxes = []
            for sl in ndimage.find_objects(labels):
                if sl is None:
                    continue
                ys, xs = sl
                box = (max(0, xs.start-1), max(0, ys.start-1),
                       min(img.width, xs.stop+1), min(img.height, ys.stop+1))
                crop = alpha[box[1]:box[3], box[0]:box[2]]
                if crop.sum() >= 2 and crop.size >= 4:
                    boxes.append(box)
            if len(boxes) > 1:
                return boxes, "alpha_components"
    return [], "unresolved"


def normalized(tile, target=32):
    tile = tile.convert("RGBA")
    bbox = tile.getbbox()
    if bbox is None:
        return None
    alpha = tile.getchannel("A")
    if sum(alpha.getdata()) < 255 * 2:
        return None
    rgb = Image.new("RGBA", tile.size, (0, 0, 0, 0)); rgb.alpha_composite(tile)
    stat = ImageStat.Stat(rgb.convert("RGB"))
    if max(stat.var) < 0.5:
        return None
    tile = tile.crop(bbox)
    if tile.size != (target, target):
        # Deliberately pass through a low-resolution bottleneck before final upsampling.
        if tile.width > target or tile.height > target:
            ratio = tile.width / max(1, tile.height)
            low = 16 if 0.5 <= ratio <= 2.0 else target
            scale0 = min(low / tile.width, low / tile.height)
            tile = tile.resize((max(1, round(tile.width*scale0)), max(1, round(tile.height*scale0))), Image.Resampling.NEAREST)
        scale = min(target / tile.width, target / tile.height)
        size = (max(1, round(tile.width * scale)), max(1, round(tile.height * scale)))
        tile = tile.resize(size, Image.Resampling.NEAREST)
        canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
        canvas.alpha_composite(tile, ((target-size[0])//2, (target-size[1])//2))
        tile = canvas
    return tile


def image_items(root):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            yield path, path.as_posix(), path.read_bytes()
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as z:
                    for info in z.infolist():
                        ext = Path(info.filename).suffix.lower()
                        if not info.is_dir() and ext in IMAGE_EXTS and info.file_size <= 100 * 1024 * 1024:
                            yield path, f"{path.as_posix()}::{info.filename}", z.read(info)
            except (zipfile.BadZipFile, OSError):
                continue


def run(root, out, target=32, max_samples=0):
    root, out = Path(root), Path(out)
    tiles_dir = out / "tiles"; tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    seen, written, sources, errors = set(), 0, 0, 0
    with manifest.open("w", encoding="utf-8") as mf:
        for container, source, data in image_items(root):
            try:
                img = Image.open(io.BytesIO(data)); img.seek(0); img = img.convert("RGBA")
            except (UnidentifiedImageError, OSError, ValueError):
                errors += 1; continue
            boxes, method = content_boxes(img, source)
            if not boxes:
                continue
            sources += 1
            for box in boxes:
                tile = normalized(img.crop(box), target)
                if tile is None: continue
                raw = tile.tobytes(); digest = hashlib.sha256(raw).hexdigest()
                if digest in seen: continue
                seen.add(digest)
                rel = Path(digest[:2]) / f"{digest}.png"; dst = tiles_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True); tile.save(dst, optimize=True)
                mf.write(json.dumps({"path": str(dst), "source": source, "container": str(container),
                                     "box": box, "method": method, "target": target}, ensure_ascii=False) + "\n")
                written += 1
                if written % 1000 == 0: print(f"tiles={written} sources={sources} errors={errors}", flush=True)
                if max_samples and written >= max_samples:
                    print(f"done tiles={written} sources={sources} errors={errors}"); return
    print(f"done tiles={written} sources={sources} errors={errors}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/itch_pixel")
    p.add_argument("--out", default="data/processed/itch_pixel32")
    p.add_argument("--target", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=0)
    a = p.parse_args(); run(a.root, a.out, a.target, a.max_samples)
