import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def phash(path, hash_size=8):
    try:
        import imagehash
    except ImportError:
        return None
    return imagehash.phash(Image.open(path), hash_size=hash_size)


def lowres_l2(path, size=16):
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.NEAREST), dtype=np.float32)
    return arr


def dedupe_records(records, exact=True, near=True, phash_threshold=4, l2_threshold=8.0):
    exact_seen = {}
    near_hashes = {}
    near_l2 = []
    kept = []
    for r in records:
        p = Path(r["path"])
        if not p.exists():
            continue
        if exact:
            h = sha256_bytes(p.read_bytes())
            if h in exact_seen:
                continue
            exact_seen[h] = r
            r["sha256"] = h
        keep_near = True
        if near:
            ph = phash(p)
            if ph is not None:
                for other_h, other_r in near_hashes.items():
                    if ph - other_h <= phash_threshold:
                        keep_near = False
                        break
            else:
                a = lowres_l2(p)
                dup = False
                for arr, other_r in near_l2:
                    if np.abs(a - arr).mean() <= l2_threshold:
                        dup = True
                        break
                if dup:
                    keep_near = False
        if not keep_near:
            continue
        if near:
            ph = phash(p)
            if ph is not None:
                near_hashes[ph] = r
            else:
                near_l2.append((lowres_l2(p), r))
        kept.append(r)
    return kept


def dedupe_manifest(manifest_path, out_path=None):
    if not Path(manifest_path).exists():
        print(f"manifest not found: {manifest_path}")
        return []
    records = [json.loads(line) for line in open(manifest_path, encoding="utf-8") if line.strip()]
    kept = dedupe_records(records)
    print(f"manifest {len(records)} -> kept {len(kept)}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dedupe_manifest(args.manifest, args.out or None)


if __name__ == "__main__":
    main()