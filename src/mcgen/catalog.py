"""Versioned catalogs and leakage audits; legacy transitive splitting for ingestion."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .config import fingerprint

ROLES = ("train", "val", "test")


def boolean(value):
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    if value is None or value == "unknown":
        return None
    raise ValueError(f"invalid tri-state boolean: {value!r}")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assign_splits(records, seed=0, fractions=(0.9, 0.05, 0.05)):
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("three positive split fractions must sum to 1")
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for r in records:
        if not isinstance(r.get("id"), str) or not isinstance(r.get("prompt"), str):
            raise ValueError("sample id and prompt must be strings")
        if not r.get("lineage") or not r.get("target_hash"):
            raise ValueError("every record requires lineage and target_hash")
        a, b = find("p:" + r["lineage"]), find("c:" + r["target_hash"])
        parents[max(a, b)] = min(a, b)
    for r in records:
        group = find("p:" + r["lineage"])
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
        r["split"] = ROLES[0 if value < fractions[0] else 1 if value < sum(fractions[:2]) else 2]
        r["split_group"] = group
    return audit(records)


def audit(records):
    seen_ids = set()
    owners = {key: {} for key in ("lineage", "target_hash", "split_group")}
    counts, targets = Counter(), {s: set() for s in ROLES}
    for r in records:
        if r["id"] in seen_ids:
            raise ValueError(f"duplicate sample id: {r['id']}")
        seen_ids.add(r["id"])
        role = r["split"]
        if role not in ROLES:
            raise ValueError(f"invalid split: {role}")
        if type(r["tileable"]) not in (bool, type(None)):
            raise ValueError("tileable must be boolean or null, not a string")
        if r["asset_type"] not in ("item", "tile", "other"):
            raise ValueError("invalid asset_type")
        for key, mapping in owners.items():
            value = r[key]
            if not value:
                raise ValueError(f"missing {key}")
            old = mapping.setdefault(value, role)
            if old != role:
                raise ValueError(f"{key} leaks across {old}/{role}: {value}")
        counts[role] += 1
        targets[role].add(r["target_hash"])
    return {"records": len(records), "split_counts": dict(counts),
            "unique_targets": {k: len(v) for k, v in targets.items()},
            "lineages": len(owners["lineage"]), "components": len(owners["split_group"])}


def write_catalog(directory, records, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    audit_report = audit(records)
    manifest = directory / "samples.jsonl"
    with manifest.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    sources = {}
    for r in records:
        for name in ("target", "reference"):
            spec = r.get(name)
            if spec:
                sources[spec["path"]] = None
    sources = {p: {"bytes": Path(p).stat().st_size, "sha256": file_hash(p)} for p in sorted(sources)}
    metadata = {"schema_version": 2, "representation": "straight_rgba_uint8",
                "native_size": provenance["native_size"], "sources": sources,
                "manifest_sha256": file_hash(manifest), "provenance": provenance,
                "audit": audit_report}
    metadata["fingerprint"] = fingerprint(metadata)
    atomic_json(directory / "catalog.json", metadata)
    return metadata


def read_catalog(directory, verify_sources=True):
    directory = Path(directory)
    meta = json.loads((directory / "catalog.json").read_text())
    if meta["schema_version"] != 2 or meta["representation"] != "straight_rgba_uint8":
        raise ValueError("unsupported catalog representation/version")
    expected = meta["fingerprint"]
    if fingerprint({k: v for k, v in meta.items() if k != "fingerprint"}) != expected:
        raise ValueError("catalog metadata fingerprint mismatch")
    if file_hash(directory / "samples.jsonl") != meta["manifest_sha256"]:
        raise ValueError("catalog manifest changed; rebuild and version the catalog")
    for path, spec in meta["sources"].items():
        if Path(path).stat().st_size != spec["bytes"]:
            raise ValueError(f"source size changed: {path}")
        if verify_sources and file_hash(path) != spec["sha256"]:
            raise ValueError(f"source content changed: {path}")
    with (directory / "samples.jsonl").open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    audit(records)
    for r in records:
        if not r.get("target"):
            raise ValueError("every record requires a target")
        for key in ("target", "reference"):
            spec = r.get(key)
            if spec is None:
                continue
            shape = spec.get("shape", [])
            if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape) or shape[-1] != 4:
                raise ValueError("mmap shape must be positive [N,H,W,4]")
            if type(spec["index"]) is not int or not 0 <= spec["index"] < shape[0]:
                raise ValueError("mmap row out of range")
            source = meta["sources"].get(spec["path"])
            if source is None or source["bytes"] != shape[0] * shape[1] * shape[2] * shape[3]:
                raise ValueError("unregistered source or mmap shape mismatch")
    return records, meta
