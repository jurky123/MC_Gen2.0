"""Group-aware splitting and leakage auditing.

Splits must be assigned at group level (project / lineage), never row-random,
and every derived split must inherit a single global assignment (P0-2/P0-3 in
docs/MC-Gen2_HD-to-MC_Design_v1.0.md).
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def group_of(record, keys):
    if isinstance(keys, str):
        keys = [keys]
    values = []
    for k in keys:
        v = record.get(k) if isinstance(record, dict) else None
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return None
        values.append(str(v))
    return "|".join(values)


def _role_for_group(group, seed, fracs):
    digest = hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8]
    value = int(digest, 16) % 10_000
    total = sum(fracs)
    pct_train = 10_000 * fracs[0] / total
    pct_val = 10_000 * (fracs[0] + fracs[1]) / total
    return "train" if value < pct_train else "val" if value < pct_val else "test"


def group_split(records, group_keys=("project_id",), seed=0,
                fracs=(0.9, 0.05, 0.05), fallback_keys=("file_name",)):
    """Assign every record a split by hashing its group keys.

    Records missing all group keys fall back to ``fallback_keys`` (still
    deterministic); if even those are missing the record is assigned
    individually by row index and reported in the audit.
    """
    groups, fallback_used = {}, 0
    for i, rec in enumerate(records):
        g = group_of(rec, group_keys)
        if g is None:
            g = group_of(rec, fallback_keys)
            if g is None:
                g = f"__row__{i}"
                fallback_used += 1
        groups[i] = g
    roles = {i: _role_for_group(g, seed, fracs) for i, g in groups.items()}
    audit = {
        "group_keys": list(group_keys),
        "seed": seed,
        "fracs": list(fracs),
        "n_groups": len(set(groups.values())),
        "n_rows": len(records),
        "fallback_rows": fallback_used,
        "split_sizes": dict(Counter(roles.values())),
    }
    return roles, audit


def check_disjoint(split_sets, name="splits"):
    """Assert the given {name: index-set} mapping is pairwise disjoint."""
    names = list(split_sets)
    clashes = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = split_sets[a] & split_sets[b]
            if inter:
                clashes[f"{a}&{b}"] = len(inter)
    if clashes:
        raise AssertionError(f"{name} leak: {clashes}")
    return True


def audit_no_leakage(holdout_splits, replay_splits, label):
    """Assert val/test rows of ``holdout_splits`` are absent from train of ``replay_splits``."""
    bad = {}
    for role in ("val", "test"):
        inter = set(holdout_splits.get(role, [])) & set(replay_splits.get("train", []))
        if inter:
            bad[f"{role}"] = len(inter)
    if bad:
        raise AssertionError(
            f"split leakage ({label}): holdout val/test rows found in replay train: {bad}")
    return True


def write_report(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
