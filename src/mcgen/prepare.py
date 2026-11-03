"""Project holdouts with cross-split duplicate quarantine; never mutate a catalog."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from .catalog import ROLES, atomic_json, audit, read_catalog, write_catalog
from .dataset import CatalogDataset
from .images import content_hash, native_grid


def project_holdout(records, seed=0, fractions=(.8, .1, .1), assignments=None):
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("three positive fractions must sum to 1")
    assignments = assignments or {}
    if any(v not in ROLES for v in assignments.values()):
        raise ValueError("invalid project split assignment")
    projects = {r["lineage"] for r in records}
    if set(assignments) - projects:
        raise ValueError("holdout plan contains unknown projects")
    plan = {}
    for project in sorted(projects):
        value = int(hashlib.sha256(f"{seed}:project:{project}".encode()).hexdigest()[:16], 16) / 2**64
        plan[project] = assignments.get(project, ROLES[0 if value < fractions[0] else 1 if value < sum(fractions[:2]) else 2])
    owners = defaultdict(set)
    for r in records:
        owners[r["target_hash"]].add(plan[r["lineage"]])
    conflict = {h for h, roles in owners.items() if len(roles) > 1}
    kept, quarantine = [], []
    for original in records:
        r = dict(original, split=plan[original["lineage"]], split_group="project:" + original["lineage"])
        if r["target_hash"] in conflict:
            quarantine.append(dict(r, exclusion_reason="duplicate_content_crosses_project_holdout"))
        else:
            kept.append(r)
    result = audit(kept)
    result.update({"quarantined_records": len(quarantine), "quarantined_unique_targets": len(conflict),
                   "largest_project_records": max(Counter(r["lineage"] for r in kept).values(), default=0),
                   "policy": "project_holdout_quarantine_v1"})
    return kept, quarantine, plan, result


def prepare_catalog(source, output, seed=0, assignments=None, compact=True):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("prepare output already exists")
    records, meta = read_catalog(source)
    kept, quarantine, plan, report = project_holdout(records, seed, assignments=assignments)
    if not all(report["split_counts"].get(s, 0) for s in ROLES):
        raise ValueError("project holdout has an empty split; supply an explicit project plan")
    output.mkdir(parents=True)
    atomic_json(output / "project_plan.json", plan)
    with (output / "quarantine.jsonl").open("w") as stream:
        for r in quarantine:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    if compact:
        # Validate and collapse each distinct target once at ingestion, keeping
        # provenance in source_target. References remain read-only source views.
        ds = CatalogDataset(kept, meta["native_size"], meta["native_size"], "train")
        count = len({r["target_hash"] for r in kept})
        shape = [count, meta["native_size"], meta["native_size"], 4]
        path = output / "native_targets.mmap"
        mmap = np.memmap(path, dtype=np.uint8, mode="w+", shape=tuple(shape))
        index, alpha = {}, {}
        for r in kept:
            key = r["target_hash"]
            if key not in index:
                a = native_grid(ds.array(r["target"]), meta["native_size"])
                if content_hash(a) != key:
                    raise ValueError(f"target content mismatch: {r['id']}")
                index[key] = len(index)
                mmap[index[key]] = a
                alpha[key] = "opaque" if np.all(a[..., 3] == 255) else "binary" if np.all((a[..., 3] == 0) | (a[..., 3] == 255)) else "continuous"
            r["source_target"] = r["target"]
            r["target"] = {"path": str(path), "shape": shape, "index": index[key]}
            r["alpha_mode"] = alpha[key]
        mmap.flush()
        del mmap
        report["alpha_targets"] = dict(Counter(alpha.values()))
        report["native_target_bytes"] = path.stat().st_size
    result = write_catalog(output / "catalog", kept,
                           {"native_size": meta["native_size"], "parent_fingerprint": meta["fingerprint"],
                            "split_policy": report["policy"], "seed": seed,
                            "quarantined_records": len(quarantine), "compacted": compact})
    report["catalog_fingerprint"] = result["fingerprint"]
    atomic_json(output / "preparation_report.json", report)
    return report
