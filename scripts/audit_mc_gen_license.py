import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from data.license_audit import classify

API = "https://api.modrinth.com/v2"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mc-texture-gen/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def lookup(mod_id):
    for slug in (mod_id, mod_id.replace("_", "-")):
        try:
            p = _get(f"{API}/project/{slug}")
            return {
                "project_id": p.get("id"),
                "slug": p.get("slug"),
                "title": p.get("title"),
                "license": (p.get("license") or {}).get("id", ""),
                "license_name": (p.get("license") or {}).get("name", ""),
            }
        except Exception:
            continue
    q = urllib.parse.quote(mod_id)
    try:
        d = _get(f"{API}/search?query={q}&limit=1&facets=" + urllib.parse.quote(json.dumps([["project_type:mod"]])))
        hits = d.get("hits", [])
        if hits:
            h = hits[0]
            return {
                "project_id": h.get("project_id"),
                "slug": h.get("slug"),
                "title": h.get("title"),
                "license": h.get("license", ""),
                "license_name": "",
            }
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/build/mc_b16/weak_labels.jsonl")
    ap.add_argument("--out", default="data/build/mc_b16/license_audit.jsonl")
    args = ap.parse_args()

    mods = []
    for l in open(args.labels, encoding="utf-8"):
        mods.append(json.loads(l)["mod_id"])
    mods = sorted(set(mods))
    print(f"auditing {len(mods)} mods")

    rows = []
    for i, m in enumerate(mods):
        info = lookup(m)
        lic = (info or {}).get("license", "")
        status = classify(lic)
        rows.append({"mod_id": m, "status": status, "license": lic, **({} if info is None else {k: v for k, v in info.items() if k != "license"})})
        print(f"  [{i + 1}/{len(mods)}] {m:24s} {status:6s} {lic}")
        time.sleep(0.25)

    summary = Counter(r["status"] for r in rows)
    print("summary:", dict(summary))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        if r["status"] in ("deny", "review"):
            print(f"    [{r['status']}] {r['mod_id']} {r['license']}")


if __name__ == "__main__":
    main()