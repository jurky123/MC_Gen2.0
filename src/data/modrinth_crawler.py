import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.modrinth.com/v2"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mc-texture-gen/0.1"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def search_projects(query, project_types=("resourcepack", "mod"), licenses=("cc0-1.0", "cc-by-4.0"), limit=20):
    type_facet = "[" + ",".join(f'"project_type:{t}"' for t in project_types) + "]"
    license_facet = "[" + ",".join(f'"license:{l}"' for l in licenses) + "]"
    facets = json.dumps([json.loads(type_facet), json.loads(license_facet)], separators=(",", ":"))
    url = f"{API}/search?query={query}&limit={limit}&index=relevance&facets={urllib.parse.quote(facets)}"
    data = _get(url)
    return data.get("hits", [])


def get_versions(project_id, version_type="release"):
    data = _get(f"{API}/project/{project_id}/version")
    for v in data:
        if v["version_type"] == version_type:
            return v
    return data[0] if data else None


def download_version(url, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "mc-texture-gen/0.1"})
    with urllib.request.urlopen(req) as r, open(out, "wb") as f:
        f.write(r.read())
    return out


def crawl(query, out_dir, limit=20, dry_run=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = search_projects(query, limit=limit)
    manifest = []
    for h in hits:
        pid = h["project_id"]
        lic = (h.get("license") or {}).get("id", "")
        ver = get_versions(pid)
        if ver is None:
            continue
        url = ver.get("files", [{}])[0].get("url")
        if not url:
            continue
        dst = out_dir / f"{pid}.zip"
        rec = {
            "project_id": pid,
            "slug": h.get("slug"),
            "title": h.get("title"),
            "license_id": lic,
            "version": ver["version_number"],
            "file_url": url,
        }
        manifest.append(rec)
        if not dry_run:
            download_version(url, dst)
            print(f"  downloaded {pid} {ver['version_number']}")
    with open(out_dir / "crawl_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"crawled {len(manifest)} projects -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--out", default="data/raw/mc/downloads")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import urllib.parse  # noqa: PLC0415

    crawl(args.query, args.out, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()