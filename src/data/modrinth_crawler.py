import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.modrinth.com/v2"


def _get(url, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mc-texture-gen/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _fetch(url, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mc-texture-gen/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    raise last


def search_projects(query, project_types=("resourcepack", "mod"), licenses=("CC0-1.0", "CC-BY-4.0"), categories=None, limit=20, sort="relevance", offset=0):
    type_facet = "[" + ",".join(f'"project_type:{t}"' for t in project_types) + "]"
    license_facet = "[" + ",".join(f'"license:{l}"' for l in licenses) + "]"
    facets = [json.loads(type_facet), json.loads(license_facet)]
    if categories:
        facets.append([f"categories:{c}" for c in categories])
    facets_str = json.dumps(facets, separators=(",", ":"))
    url = f"{API}/search?query={urllib.parse.quote(query)}&limit={limit}&offset={offset}&index={sort}&facets={urllib.parse.quote(facets_str)}"
    data = _get(url)
    return data.get("hits", []), data.get("total_hits", 0)


def get_versions(project_id, version_type="release"):
    data = _get(f"{API}/project/{project_id}/version")
    for v in data:
        if v["version_type"] == version_type:
            return v
    return data[0] if data else None


def download_version(url, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _fetch(url)
    with open(out, "wb") as f:
        f.write(data)
    return out


def _latest_file(h):
    pid = h["project_id"]
    lic = h.get("license") or ""
    ver = get_versions(pid)
    if ver is None:
        return None
    files = ver.get("files", [])
    if not files:
        return None
    return {
        "project_id": pid,
        "slug": h.get("slug"),
        "title": h.get("title"),
        "license_id": lic,
        "version": ver["version_number"],
        "version_id": ver.get("id"),
        "file_url": files[0].get("url"),
        "file_name": files[0].get("filename"),
        "file_size": files[0].get("size"),
    }


def crawl(query, out_dir, limit=20, dry_run=False, sort="relevance"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits, total = search_projects(query, limit=limit, sort=sort)
    manifest = []
    for h in hits:
        rec = _latest_file(h)
        if rec is None:
            continue
        manifest.append(rec)
        if not dry_run and rec.get("file_url"):
            dst = out_dir / f"{rec['project_id']}.zip"
            if not dst.exists():
                try:
                    download_version(rec["file_url"], dst)
                    print(f"  downloaded {rec['project_id']} {rec['version']}")
                except Exception as e:
                    print(f"  failed {rec['project_id']}: {e}")
    with open(out_dir / "crawl_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"crawled {len(manifest)}/{total} projects -> {out_dir}")
    return manifest


def crawl_all(project_types, licenses, out_dir, max_projects=200, dry_run=False, sort="downloads", categories=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    offset = 0
    while len(manifest) < max_projects:
        hits, total = search_projects("", project_types=project_types, licenses=licenses, categories=categories, limit=100, sort=sort, offset=offset)
        if not hits:
            break
        for h in hits:
            pid = h["project_id"]
            if pid in seen:
                continue
            seen.add(pid)
            rec = _latest_file(h)
            if rec is None:
                continue
            manifest.append(rec)
            if not dry_run and rec.get("file_url"):
                dst = out_dir / f"{rec['project_id']}.zip"
                if not dst.exists():
                    try:
                        download_version(rec["file_url"], dst)
                        print(f"  downloaded {pid} {rec['version']} ({rec.get('file_size', 0) // 1000}KB)")
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"  failed {pid}: {e}")
            if len(manifest) >= max_projects:
                break
        offset += 100
        with open(out_dir / "crawl_manifest.jsonl", "w", encoding="utf-8") as f:
            for r in manifest:
                f.write(json.dumps(r) + "\n")
        print(f"  scanned {offset}/{total}, collected {len(manifest)}")
        if offset >= total:
            break
    with open(out_dir / "crawl_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"collected {len(manifest)} projects -> {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--out", default="data/raw/mc/downloads")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sort", default="relevance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    crawl(args.query, args.out, limit=args.limit, dry_run=args.dry_run, sort=args.sort)


if __name__ == "__main__":
    main()