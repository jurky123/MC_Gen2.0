"""Normalize already-individual images stored in Parquet and retain text metadata."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError



def normalize_individual(image: Image.Image, target: int) -> Image.Image | None:
    """Normalize a known single image without rejecting valid low-variance textures."""
    tile = image.convert("RGBA")
    bbox = tile.getbbox()
    if bbox is None:
        return None
    tile = tile.crop(bbox)
    scale = min(target / tile.width, target / tile.height)
    size = (max(1, round(tile.width * scale)), max(1, round(tile.height * scale)))
    if tile.size != size:
        tile = tile.resize(size, Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
    canvas.alpha_composite(tile, ((target - size[0]) // 2, (target - size[1]) // 2))
    return canvas


def infer_split(path: Path) -> str:
    name = path.name.lower()
    for split in ("validation", "train", "test"):
        if split in name:
            return split
    return "train"


def serializable_metadata(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if key == "image" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
    return result


def run(root: Path, out: Path, target: int = 32) -> None:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")

    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    seen = set()
    written = duplicates = errors = rows = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for parquet_path in files:
            split = infer_split(parquet_path)
            parquet = pq.ParquetFile(parquet_path)
            offset = 0
            for batch in parquet.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    row_index = offset
                    offset += 1
                    rows += 1
                    image_value = row.get("image")
                    data = image_value.get("bytes") if isinstance(image_value, dict) else None
                    if not data:
                        errors += 1
                        continue
                    try:
                        image = Image.open(io.BytesIO(data)).convert("RGBA")
                        tile = normalize_individual(image, target)
                    except (UnidentifiedImageError, OSError, ValueError):
                        errors += 1
                        continue
                    if tile is None:
                        errors += 1
                        continue
                    digest = hashlib.sha256(tile.tobytes()).hexdigest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    rel = Path(digest[:2]) / f"{digest}.png"
                    destination = tiles_dir / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    tile.save(destination, optimize=True)
                    record = {
                        "path": str(destination),
                        "source": f"{parquet_path.as_posix()}#{row_index}",
                        "container": str(parquet_path),
                        "split": split,
                        "method": "individual_alpha_bbox",
                        "target": target,
                        "sha256_rgba": digest,
                        "metadata": serializable_metadata(row),
                    }
                    record["weak_prompt"] = str(
                        row.get("text")
                        or row.get("overall_texture_description")
                        or row.get("texture_name")
                        or "pixel art game asset"
                    )
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 1000 == 0:
                        print(f"written={written} rows={rows} duplicates={duplicates} errors={errors}", flush=True)

    summary = {"rows": rows, "written": written, "duplicates": duplicates, "errors": errors}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("done " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=32)
    args = parser.parse_args()
    run(args.root, args.out, args.target)
