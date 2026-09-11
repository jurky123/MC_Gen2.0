import argparse
import json

ALLOW = {"cc0-1.0", "cc-by-4.0"}
REVIEW = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "epl-2.0",
    "artistic-2.0",
}
DENY_SUBSTRINGS = [
    "arr",
    "all-rights-reserved",
    "reserved",
    "no-ai",
    "ai-training",
    "prohibited",
    "custom",
    "unclear",
    "mixed",
    "proprietary",
    "licenseref",
    "noncommercial",
    "non-commercial",
    "nc-",
]


def classify(license_id, extra="", allow_override=None, review_override=None, deny_override=None):
    allow = set(allow_override or ALLOW)
    review = set(review_override or REVIEW)
    deny = set(deny_override or DENY_SUBSTRINGS)
    lid = (license_id or "").lower().strip().replace(" ", "-")
    if lid in allow:
        return "allow"
    if lid in review:
        return "review"
    if any(k in lid for k in deny):
        return "deny"
    blob = f"{lid} {extra}".lower()
    if any(k in blob for k in ["no-ai", "ai-training", "prohibited", "arr"]):
        return "deny"
    return "review"


def audit_records(records, allow_override=None, review_override=None, deny_override=None):
    summary = {"allow": 0, "review": 0, "deny": 0}
    out = {"allow": [], "review": [], "deny": []}
    for r in records:
        status = classify(
            r.get("license_id", ""),
            extra=r.get("license_extra", ""),
            allow_override=allow_override,
            review_override=review_override,
            deny_override=deny_override,
        )
        summary[status] += 1
        out[status].append(r)
    return summary, out


def audit_manifest(manifest_path):
    records = [json.loads(line) for line in open(manifest_path, encoding="utf-8") if line.strip()]
    summary, grouped = audit_records(records)
    return summary, grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()
    summary, grouped = audit_manifest(args.manifest)
    print(json.dumps(summary, indent=2))
    for status in ("deny", "review"):
        for r in grouped[status][:10]:
            print(f"  [{status}] {r.get('project_id')} {r.get('license_id')}")


if __name__ == "__main__":
    main()