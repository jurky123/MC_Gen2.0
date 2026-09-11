import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = "gvecchio/MatSynth"
COLS = ["name", "category", "metadata", "basecolor"]
ALLOW_LICENSES = {"CC0", "CC-BY-4.0", "CC-BY"}
TARGET = 1024
UA = "mc-texture-gen/0.1"
CHUNK = 8 * 1024 * 1024
DEFAULT_ENDPOINTS = ["https://hf-mirror.com", "https://hf-mirror.net"]


def _endpoints():
    env = os.environ.get("HF_ENDPOINT", "")
    if env:
        return [e.strip().rstrip("/") for e in env.split(",") if e.strip()]
    return DEFAULT_ENDPOINTS


def _hf_url(base, path):
    return f"{base}/datasets/{REPO}/resolve/main/{path}"


class HTTPRangeFile:
    def __init__(self, url, timeout=120, retries=5):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.length = self._fetch_length()

    def _range(self, start, end):
        last = None
        for i in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end - start + 1}")
                return data
            except Exception as e:
                last = e
                time.sleep(2 * (i + 1))
        raise last

    def _fetch_length(self):
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            cr = r.headers.get("Content-Range", "")
            r.read()
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        raise IOError("server does not support range requests")

    def read(self, n=-1):
        if n < 0:
            out = bytearray()
            while self.pos < self.length:
                end = min(self.pos + CHUNK - 1, self.length - 1)
                out += self._range(self.pos, end)
                self.pos = end + 1
            return bytes(out)
        if n == 0:
            return b""
        end = min(self.pos + n - 1, self.length - 1)
        data = self._range(self.pos, end)
        self.pos = end + 1
        return data

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.length + offset
        else:
            raise ValueError(whence)
        self.pos = max(0, min(self.pos, self.length))
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        pass

    def seekable(self):
        return True


def _decode_image(cell):
    if cell is None:
        return None
    b = cell.get("bytes") if isinstance(cell, dict) else cell
    if b is None:
        return None
    img = Image.open(io.BytesIO(b))
    img = img.convert("RGB")
    if img.size != (TARGET, TARGET):
        img = img.resize((TARGET, TARGET), Image.Resampling.LANCZOS)
    return img


def _list_shards():
    shards = []
    shards += [f"data/train-{i:05d}-of-00431.parquet" for i in range(431)]
    shards += [f"data/test-{i:05d}-of-00008.parquet" for i in range(8)]
    return shards


def _load_done(meta_path):
    if not Path(meta_path).exists():
        return set()
    return {json.loads(l)["name"] for l in open(meta_path, encoding="utf-8") if l.strip()}


def _load_done_shards(path):
    if not Path(path).exists():
        return set()
    return {l.strip() for l in open(path, encoding="utf-8") if l.strip()}


def run(out_dir, licenses=None, max_shards=None, max_materials=None, split_filter=None, max_fail=8):
    licenses = set(licenses or ALLOW_LICENSES)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "matsynth_metadata.jsonl"
    done_shards_path = out_dir / "done_shards.txt"
    done = _load_done(meta_path)
    done_shards = _load_done_shards(done_shards_path)

    shards = _list_shards()
    if split_filter and split_filter != "all":
        shards = [s for s in shards if split_filter in s.split("/")[1].split("-")[0]]
    if max_shards:
        shards = shards[:max_shards]
    print(f"processing {len(shards)} shards")

    n = 0
    fails = 0
    meta_f = open(meta_path, "a", encoding="utf-8")
    done_f = open(done_shards_path, "a", encoding="utf-8")
    for si, shard in enumerate(shards):
        if shard in done_shards:
            continue
        attempt = 0
        while True:
            attempt += 1
            try:
                f = None
                for base in _endpoints():
                    try:
                        f = HTTPRangeFile(_hf_url(base, shard))
                        break
                    except Exception as e:
                        print(f"  open {base} failed: {type(e).__name__}")
                        time.sleep(2)
                if f is None:
                    raise IOError("all endpoints failed to open")
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    t = pf.read_row_group(i, columns=COLS)
                    for r in range(t.num_rows):
                        md = t.column("metadata")[r].as_py() or {}
                        if md.get("license", "") not in licenses:
                            continue
                        name = t.column("name")[r].as_py() or f"mat_{n}"
                        if name in done:
                            continue
                        img = _decode_image(t.column("basecolor")[r].as_py())
                        if img is None:
                            continue
                        dst = out_dir / name / f"{name}_Color.jpg"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        img.save(dst, "JPEG", quality=95)
                        rec = {
                            "name": name,
                            "license": md.get("license"),
                            "source": md.get("source"),
                            "method": md.get("method"),
                            "category": md.get("category"),
                            "tags": md.get("tags", []),
                            "stationary": md.get("stationary"),
                        }
                        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        meta_f.flush()
                        done.add(name)
                        n += 1
                        if max_materials and n >= max_materials:
                            break
                    if max_materials and n >= max_materials:
                        break
                f.close()
                done_f.write(shard + "\n")
                done_f.flush()
                print(f"  [{si + 1}/{len(shards)}] {shard} done -> total {n} saved")
                break
            except Exception as e:
                wait = min(60, 4 * attempt)
                print(f"  [{si + 1}/{len(shards)}] {shard} attempt {attempt} failed ({type(e).__name__}): {str(e)[:80]}; retry in {wait}s")
                time.sleep(wait)
            if attempt >= 60:
                print(f"  giving up shard {shard} after {attempt} attempts")
                break
        if max_materials and n >= max_materials:
            break
    meta_f.close()
    done_f.close()
    print(f"matsynth: saved {n} materials -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/matsynth")
    ap.add_argument("--licenses", nargs="*", default=["CC0"])
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--max-materials", type=int, default=0)
    ap.add_argument("--split", default="all")
    ap.add_argument("--max-fail", type=int, default=8)
    ap.add_argument("--token", default="")
    args = ap.parse_args()
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    run(
        args.out,
        licenses=args.licenses,
        max_shards=args.max_shards or None,
        max_materials=args.max_materials or None,
        split_filter=args.split,
        max_fail=args.max_fail,
    )


if __name__ == "__main__":
    main()