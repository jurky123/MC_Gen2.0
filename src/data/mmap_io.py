"""Unified headerless raw mmap I/O.

All large image/text arrays in this repo are stored as headerless raw mmap
files whose shape must be recovered from metadata (row count) + file size.
``np.lib.format.open_memmap`` (used previously in build_mmap) writes a NPY
header that the reader does not skip, so it is banned here; every writer and
reader must go through this module (P0-1 in docs/MC-Gen2_HD-to-MC_Design_v1.0.md).
"""
import json
import os
from pathlib import Path

import numpy as np

MAGIC = "mcgen_raw_mmap_v1"


def write_raw_mmap(array, path, dtype=None):
    """Dump a contiguous array as headerless raw bytes (atomic replace)."""
    arr = np.ascontiguousarray(array, dtype=dtype or array.dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        arr.tofile(handle)
    os.replace(tmp, path)
    return path


def read_raw_mmap(path, shape, dtype=np.uint8):
    """Memory-map a headerless raw file with an explicit shape."""
    path = Path(path)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(f"{path}: {actual} bytes, expected {expected} for shape {shape} {dtype}")
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def infer_n_rows(path, item_shape, dtype=np.uint8):
    """Row count of a headerless raw mmap given one row's shape."""
    item = int(np.prod(item_shape)) * np.dtype(dtype).itemsize
    total = os.path.getsize(path)
    if item <= 0 or total % item != 0:
        raise ValueError(f"{path}: {total} bytes is not a multiple of row size {item}")
    return total // item


def write_schema(out_dir, name, shape, dtype, extra=None):
    """Write schema.json describing mmap files in a build directory."""
    payload = {
        "format": MAGIC,
        "files": {name: {"shape": list(shape), "dtype": str(np.dtype(dtype))}},
    }
    if extra:
        payload.update(extra)
    out = Path(out_dir)
    existing = out / "schema.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        old.setdefault("files", {}).update(payload["files"])
        old["format"] = MAGIC
        payload = old
    existing.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return existing


def read_schema(out_dir):
    return json.loads((Path(out_dir) / "schema.json").read_text(encoding="utf-8"))
