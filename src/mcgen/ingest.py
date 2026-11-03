"""Future data ingestion through an explicit image manifest, not legacy builders."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .catalog import assign_splits, boolean, read_catalog, write_catalog, file_hash
from .images import content_hash, decode, native_grid, reference


def import_images(manifest, output, size=16, ref_size=128, seed=0):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or size not in (8, 16, 32, 64) or ref_size < size:
        raise ValueError("nonempty manifest and valid native/reference sizes required")
    output.mkdir(parents=True, exist_ok=False)
    n = len(rows)
    target_path, ref_path = output / "targets.mmap", output / "references.mmap"
    targets = np.memmap(target_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 4))
    refs = np.memmap(ref_path, mode="w+", dtype=np.uint8, shape=(n, ref_size, ref_size, 4))
    records = []
    for i, row in enumerate(rows):
        for key in ("id", "lineage", "prompt", "asset_type", "target", "tileable"):
            if key not in row:
                raise ValueError(f"row {i} missing {key}")
        target_file = manifest.parent / row["target"]
        with Image.open(target_file) as im:
            a = native_grid(im.convert("RGBA"), size)
        targets[i] = a
        ref = None
        if row.get("reference"):
            with Image.open(manifest.parent / row["reference"]) as im:
                refs[i] = decode(reference(im.convert("RGBA"), ref_size))
            ref = {"path": str(ref_path), "shape": [n, ref_size, ref_size, 4], "index": i}
        records.append({"id": row["id"], "lineage": row["lineage"], "prompt": row["prompt"],
                        "asset_type": row["asset_type"], "tileable": boolean(row["tileable"]),
                        "source_license": row.get("license", "unknown"), "generator": row.get("generator", "external"),
                        "target_hash": content_hash(a), "style": row.get("style", ""),
                        "alpha_mode": "opaque" if np.all(a[..., 3] == 255) else "binary" if np.all((a[..., 3] == 0) | (a[..., 3] == 255)) else "continuous",
                        "target": {"path": str(target_path), "shape": [n, size, size, 4], "index": i},
                        "reference": ref})
    targets.flush()
    refs.flush()
    del targets, refs
    assign_splits(records, seed)
    return write_catalog(output / "catalog", records, {"native_size": size, "reference_size": ref_size,
                         "source_manifest": str(manifest), "manifest_sha256": file_hash(manifest), "seed": seed})


def merge_catalogs(catalogs, output, seed=0):
    all_records, parents, size = [], [], None
    for path in catalogs:
        records, meta = read_catalog(path)
        if size is not None and size != meta["native_size"]:
            raise ValueError("cannot merge different native grids")
        size = meta["native_size"]
        all_records.extend(records)
        parents.append(meta["fingerprint"])
    if not all_records:
        raise ValueError("no records to merge")
    # Existing split labels are deliberately replaced. The new identity makes
    # old checkpoint resume fail; a merged corpus must not inherit two registries.
    assign_splits(all_records, seed)
    return write_catalog(output, all_records, {"native_size": size, "parent_catalogs": parents, "seed": seed})
