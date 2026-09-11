"""Convert the NathMen12 parquet dataset to the project's raw mmap format."""
import argparse, hashlib, io, json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

META_COLS = ["file_name", "type", "project_id", "mod_slug", "author", "license", "project_type", "version_url", "text"]


def split_name(project_id):
    value = int(hashlib.sha1(str(project_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 90 else "val" if value < 95 else "test"


def build(source, out, image_size=32, channels=4):
    files = sorted(Path(source).glob("*.parquet")); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    images = np.memmap(out / "images.uint8.mmap", dtype=np.uint8, mode="w+", shape=(total, image_size, image_size, channels))
    writer = None; splits = {"train": [], "val": [], "test": []}; offset = 0
    for file in files:
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=4096, columns=["image"] + META_COLS):
            rows = batch.to_pylist(); meta = []
            for row in rows:
                cell = row.pop("image") or {}; raw = cell.get("bytes")
                img = Image.open(io.BytesIO(raw)).convert("RGBA")
                if channels == 4:
                    tile = img.resize((image_size, image_size), Image.Resampling.NEAREST)
                else:
                    bg = Image.new("RGBA", img.size, (0, 0, 0, 255)); bg.alpha_composite(img)
                    tile = bg.convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
                images[offset] = np.asarray(tile, dtype=np.uint8)
                row["index"] = offset; row["weak_prompt"] = row.get("text") or ""
                splits[split_name(row.get("project_id"))].append(offset)
                meta.append(row); offset += 1
            table = pa.Table.from_pylist(meta)
            if writer is None: writer = pq.ParquetWriter(out / "metadata.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            if offset % 50000 < len(rows): print(f"wrote {offset}/{total}", flush=True)
    if writer: writer.close()
    images.flush(); del images
    with (out / "splits.json").open("w") as f: json.dump(splits, f)
    print(f"done rows={offset} channels={channels} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", default="data/raw/mc_text2image/data")
    p.add_argument("--out", default="data/build/mc_text2image32"); p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=4, choices=(3, 4))
    a = p.parse_args(); build(a.source, a.out, a.image_size, a.channels)
