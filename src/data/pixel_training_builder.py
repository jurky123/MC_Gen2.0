"""Build a unified training mmap from processed pixel manifests and mmap datasets.

Supports RGB (3) or RGBA (4) output. With ``--channels 4`` transparent pixel-art
sources keep their alpha channel instead of being composited onto black; opaque
mmap sources (e.g. Minecraft textures) get an all-opaque alpha channel.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SPLIT_NAMES = ("train", "val", "test")


def split_for(container):
    value = int(hashlib.sha1(container.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 90 else ("val" if value < 95 else "test")


def _manifest_records(manifests):
    """Yield globally deduplicated pixel records without retaining full metadata."""
    seen = set()
    for manifest in map(Path, manifests):
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rec = json.loads(line)
                    rel = str(rec["path"]).replace("\\", "/")
                    digest = Path(rel).stem
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"invalid record in {manifest}:{line_number}") from exc
                if digest in seen:
                    continue
                seen.add(digest)
                rec["source_set"] = manifest.parent.name
                rec["content_sha256"] = digest
                yield rec


def _dataset_info(dataset_dir):
    dataset_dir = Path(dataset_dir)
    metadata = dataset_dir / "metadata.parquet"
    images = dataset_dir / "images.uint8.mmap"
    splits_path = dataset_dir / "splits.json"
    for path in (metadata, images, splits_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = pq.ParquetFile(metadata).metadata.num_rows
    array_format = "raw"
    with images.open("rb") as handle:
        if handle.read(6) == b"\x93NUMPY":
            array_format = "npy"
    if array_format == "npy":
        array = np.load(images, mmap_mode="r")
        if (array.ndim != 4 or array.shape[0] != rows or array.shape[-1] not in (3, 4)
                or array.shape[1] != array.shape[2]):
            raise ValueError(f"invalid NPY image shape {array.shape} in {images}")
        size = array.shape[1]
        channels = int(array.shape[-1])
        array._mmap.close()
    else:
        total = images.stat().st_size
        size = channels = None
        for c in (4, 3, 1):
            if rows <= 0 or total % (rows * c):
                continue
            px = total // (rows * c)
            s = int(round(px ** 0.5))
            if s * s == px and s >= 8:
                size, channels = s, c
                break
        if size is None:
            raise ValueError(f"cannot infer image shape for {images} (rows={rows})")
    with splits_path.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    assigned = sum(len(splits.get(name, [])) for name in SPLIT_NAMES)
    if assigned != rows:
        raise ValueError(f"split count {assigned} != metadata rows {rows} in {dataset_dir}")
    return {"dir": dataset_dir, "metadata": metadata, "images": images,
            "rows": rows, "size": size, "channels": channels, "splits": splits,
            "array_format": array_format}


def _adapt_channels(batch, source_channels, target_channels):
    if source_channels == target_channels:
        return batch
    if source_channels == 3 and target_channels == 4:
        alpha = np.full(batch.shape[:3] + (1,), 255, dtype=np.uint8)
        return np.concatenate([batch, alpha], axis=-1)
    if source_channels == 4 and target_channels == 3:
        rgb = batch[..., :3].astype(np.uint16)
        alpha = batch[..., 3:4].astype(np.uint16)
        return ((rgb * alpha + 127) // 255).astype(np.uint8)
    raise ValueError(f"cannot adapt {source_channels}-channel batch to {target_channels} channels")


def _resize(batch, source_size, target_size, channels):
    batch = _adapt_channels(batch, batch.shape[-1], channels)
    if source_size == target_size:
        return batch
    mode = "RGBA" if channels == 4 else "RGB"
    out = np.empty((len(batch), target_size, target_size, channels), dtype=np.uint8)
    for i, array in enumerate(batch):
        out[i] = np.asarray(Image.fromarray(array, mode).resize(
            (target_size, target_size), Image.Resampling.NEAREST))
    return out


def _to_target(image, size, channels):
    image = image.convert("RGBA")
    if channels == 4:
        tile = image.resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(tile, dtype=np.uint8)
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    background.alpha_composite(image)
    rgb = background.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(rgb, dtype=np.uint8)


def _write_metadata_table(writer, records):
    rows = []
    for rec in records:
        rows.append({
            "index": int(rec.pop("index")),
            "source_set": str(rec.get("source_set", "")),
            "weak_prompt": str(rec.get("weak_prompt") or ""),
            "source_metadata": json.dumps(rec, ensure_ascii=False, default=str),
        })
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))


def build(manifests, datasets, out, size=32, chunk_size=8192, channels=4):
    manifests = [Path(p) for p in manifests]
    dataset_infos = [_dataset_info(p) for p in datasets]
    pixel_count = sum(1 for _ in _manifest_records(manifests))
    total = sum(info["rows"] for info in dataset_infos) + pixel_count
    if total == 0:
        raise ValueError("no input samples")

    out = Path(out)
    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    images = np.memmap(tmp / "images.uint8.mmap", dtype=np.uint8, mode="w+",
                       shape=(total, size, size, channels))
    splits = {name: [] for name in SPLIT_NAMES}
    schema = pa.schema([("index", pa.int64()), ("source_set", pa.string()),
                        ("weak_prompt", pa.string()), ("source_metadata", pa.string())])
    writer = pq.ParquetWriter(tmp / "metadata.parquet", schema, compression="zstd")
    offset = 0
    source_counts = {}
    try:
        for info in dataset_infos:
            source_set = info["dir"].name
            if info["array_format"] == "npy":
                src = np.load(info["images"], mmap_mode="r")
            else:
                src = np.memmap(info["images"], dtype=np.uint8, mode="r",
                                shape=(info["rows"], info["size"], info["size"], info["channels"]))
            split_lookup = {int(i): name for name in SPLIT_NAMES
                            for i in info["splits"].get(name, [])}
            parquet = pq.ParquetFile(info["metadata"])
            source_written = 0
            for batch in parquet.iter_batches(batch_size=chunk_size):
                records = batch.to_pylist()
                count = len(records)
                images[offset:offset + count] = _resize(
                    src[source_written:source_written + count], info["size"], size, channels)
                meta_rows = []
                for local, rec in enumerate(records):
                    old_index = source_written + local
                    split = split_lookup.get(old_index)
                    if split is None:
                        raise ValueError(f"index {old_index} missing from splits in {info['dir']}")
                    splits[split].append(offset + local)
                    rec.pop("index", None)
                    rec.update(index=offset + local, source_set=source_set,
                               weak_prompt=rec.get("weak_prompt") or rec.get("text") or "")
                    meta_rows.append(rec)
                _write_metadata_table(writer, meta_rows)
                offset += count
                source_written += count
                print(f"wrote {offset}/{total} ({source_set})", flush=True)
            source_counts[source_set] = source_written
            src._mmap.close()

        pixel_batch = []
        for rec in _manifest_records(manifests):
            rel = str(rec["path"]).replace("\\", "/")
            path = Path(rel)
            if not path.is_absolute():
                path = Path.cwd() / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                images[offset] = _to_target(image, size, channels)
            group = str(rec.get("container") or rec.get("source") or path)
            splits[split_for(group)].append(offset)
            rec.update(index=offset, weak_prompt=rec.get("weak_prompt") or "pixel art game asset")
            pixel_batch.append(rec)
            source_counts[rec["source_set"]] = source_counts.get(rec["source_set"], 0) + 1
            offset += 1
            if len(pixel_batch) >= chunk_size:
                _write_metadata_table(writer, pixel_batch)
                pixel_batch = []
                print(f"wrote {offset}/{total} (pixel assets)", flush=True)
        _write_metadata_table(writer, pixel_batch)
    finally:
        writer.close()
        images.flush()

    if offset != total:
        raise RuntimeError(f"wrote {offset} rows, expected {total}")
    with (tmp / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits, handle)
    summary = {"rows": total, "image_size": size, "channels": channels,
               "sources": source_counts,
               "splits": {name: len(splits[name]) for name in SPLIT_NAMES},
               "manifests": [str(p) for p in manifests], "datasets": [str(p) for p in datasets]}
    with (tmp / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    images._mmap.close()
    if out.exists():
        raise FileExistsError(f"refusing to replace existing output: {out}")
    os.replace(tmp, out)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="*", help="processed pixel manifest JSONL files")
    parser.add_argument("--dataset", action="append", default=[],
                        help="existing mmap dataset directory; may be repeated")
    parser.add_argument("--out", default="data/build/stage1_32")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4, choices=(3, 4))
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    build(args.manifests, args.dataset, args.out, args.size, args.chunk_size, args.channels)
