"""Rebuild Stage-3 / replay splits under a single global holdout (P0-3), and
re-verify the Stage-3 subset data volume (user request, 2026-09-19).

Rules enforced here:
1. The global splits.json (project-grouped, v2.3) is the ONLY holdout: its
   val/test rows are never trained on, by any stage, through any source.
2. stage3_splits.json is re-derived as an inheritance of the global split:
   subset rows are restricted to global-train rows, then project-grouped
   90/5/5. Subset rows living in global val/test are dropped (unused).
3. replay_splits.json train = global train minus (stage3 val ∪ stage3 test)
   minus pixel-duplicate rows of those holdout rows. Replay val/test mirror
   the global val/test minus all subset rows.
4. Train-time consumers must use replay_splits.json for the replay source.

Run:  python scripts/rebuild_splits.py
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.lineage_split import _role_for_group, audit_no_leakage, check_disjoint  # noqa: E402
from data.filename_prompts import clean_tokens  # noqa: E402
from scripts.select_stage3_subset import COLORS, FORMS, MATERIALS, STATES  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"
STAGE3_FRACTIONS = (0.9, 0.05, 0.05)
STAGE3_SPLIT_SEED = 20260919
ALPHA_SAMPLE = 512

# v1 heuristic: item sprites are never tileable; blocks are, except for a few
# structural families whose borders are load-bearing (doors/signs/...).
NON_TILEABLE_BLOCK_KEYWORDS = (
    "door", "trapdoor", "sign", "banner", "ladder", "boat", "minecart", "rail",
)


def tileable_column(meta):
    is_block = (meta["type"].astype(str) == "block").to_numpy()
    names = meta["file_name"].astype(str).str.lower().to_numpy()
    structural = np.array([any(k in n for k in NON_TILEABLE_BLOCK_KEYWORDS) for n in names])
    return is_block & ~structural


def load_images(build, n_rows):
    path = build / "images.uint8.mmap"
    size = int(os.path.getsize(path) // (n_rows * 4)) if False else None
    total = os.path.getsize(path)
    per_row = total // n_rows
    channels = per_row // (32 * 32)
    assert channels in (3, 4), channels
    return np.memmap(path, dtype=np.uint8, mode="r",
                     shape=(n_rows, 32, 32, channels))


def row_hashes(images, rows):
    return {i: hashlib.sha1(np.asarray(images[i]).tobytes()).hexdigest() for i in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(BUILD))
    args = ap.parse_args()
    build = Path(args.build)

    global_splits = json.loads((build / "splits.json").read_text())
    meta = pd.read_parquet(build / "metadata.parquet")
    subset_rows = sorted(int(json.loads(line)["index"])
                         for line in (build / "stage3_subset.jsonl").read_text().splitlines()
                         if line.strip())
    images = load_images(build, len(meta))

    # ---- 1. restrict the subset to global-train rows ----------------------
    global_train = set(global_splits["train"])
    usable = [i for i in subset_rows if i in global_train]
    dropped = sorted(set(subset_rows) - set(usable))

    # ---- 2. project-grouped stage3 split (inherits the global holdout) ----
    roles = {}
    for i in usable:
        roles[i] = _role_for_group(str(meta.at[i, "project_id"]), STAGE3_SPLIT_SEED, STAGE3_FRACTIONS)
    stage3_splits = {role: sorted(i for i, r in roles.items() if r == role)
                     for role in ("train", "val", "test")}
    check_disjoint({k: set(v) for k, v in stage3_splits.items()}, "stage3")
    for role, rows in stage3_splits.items():
        assert set(rows) <= global_train, f"stage3 {role} leaks outside global train"

    # ---- 3. replay train: project-clean (formal) + sample-clean (ref) ----
    # Formal standard (design review 2026-09-20): exclude every row whose
    # project contains a stage3 val/test row, then pixel dups of holdout rows.
    # Sample-clean (row + exact-dup exclusion only) is kept for reference.
    holdout = set(stage3_splits["val"]) | set(stage3_splits["test"])
    holdout_projects = set(meta.loc[sorted(holdout), "project_id"].astype(str))
    holdout_hashes = set(row_hashes(images, sorted(holdout)).values())
    proj_clean, sample_clean, dups = [], [], 0
    for i in global_splits["train"]:
        if i in holdout:
            continue
        h = hashlib.sha1(np.asarray(images[i]).tobytes()).hexdigest()
        if h in holdout_hashes:
            dups += 1
            continue
        sample_clean.append(i)
        if str(meta.at[i, "project_id"]) not in holdout_projects:
            proj_clean.append(i)
    replay_splits = {
        "train": proj_clean,
        "val": [i for i in global_splits["val"] if i not in set(subset_rows)],
        "test": [i for i in global_splits["test"] if i not in set(subset_rows)],
    }
    replay_sampleclean = {
        "train": sample_clean,
        "val": replay_splits["val"],
        "test": replay_splits["test"],
    }
    audit_no_leakage(stage3_splits, {"train": replay_splits["train"]}, "stage3 vs replay")
    check_disjoint({k: set(v) for k, v in replay_splits.items()}, "replay")

    # ---- 4. write outputs -------------------------------------------------
    (build / "stage3_splits.json").write_text(json.dumps(stage3_splits))
    (build / "replay_splits.json").write_text(json.dumps(replay_splits))
    (build / "replay_splits_sampleclean.json").write_text(json.dumps(replay_sampleclean))
    if not (build / "splits.json.bak_v23").exists():
        (build / "splits.json.bak_v23").write_text(json.dumps(global_splits, indent=2))

    report = build_subset_report(build, meta, images, stage3_splits, subset_rows,
                                 dropped=dropped, dup_rows=dups,
                                 holdout_projects=len(holdout_projects),
                                 proj_rows_removed=len(sample_clean) - len(proj_clean),
                                 replay_train_rows=len(proj_clean))
    (build / "stage3_subset_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def row_hashes(images, rows):
    return {i: hashlib.sha1(np.asarray(images[i]).tobytes()).hexdigest() for i in rows}


def build_subset_report(build, meta, images, stage3_splits, subset_rows,
                        dropped, dup_rows, holdout_projects=0, proj_rows_removed=0,
                        replay_train_rows=0):
    subset_df = meta.loc[sorted(subset_rows)]
    usable_rows = stage3_splits["train"] + stage3_splits["val"] + stage3_splits["test"]
    usable_df = meta.loc[sorted(usable_rows)]
    per_project = Counter(usable_df["project_id"].astype(str))
    token_sets = [set(clean_tokens(str(f), keep_parts=True, keep_anim=True, keep_generic=False))
                  for f in subset_df["file_name"]]
    coverage = {}
    for name, vocab in (("forms", FORMS), ("materials", MATERIALS),
                        ("colours", COLORS), ("states", STATES)):
        covered = {w for w in vocab if any(w in t for t in token_sets)}
        coverage[name] = {"vocab": len(vocab), "covered": len(covered),
                          "rows": sum(1 for t in token_sets if t & vocab)}
    sample = np.asarray(images[sorted(usable_rows)[:ALPHA_SAMPLE]])
    return {
        "subset_size": len(subset_rows),
        "usable_size": len(usable_rows),
        "dropped_rows_outside_global_train": len(dropped),
        "splits": {k: len(v) for k, v in stage3_splits.items()},
        "block_item": dict(Counter(subset_df["type"].astype(str))),
        "n_projects": len(per_project),
        "per_project_max": int(max(per_project.values())),
        "coverage": coverage,
        "alpha": {
            "sampled": int(len(sample)),
            "transparent_fraction": float((sample[..., 3] < 250).mean()),
            "fully_transparent_fraction": float((sample[..., 3] == 0).mean()),
        },
        "duplicate_replay_rows_removed": int(dup_rows),
        "holdout_projects": int(holdout_projects),
        "project_rows_removed_from_replay": int(proj_rows_removed),
        "replay_train_rows_project_clean": int(replay_train_rows),
        "leakage_check": "stage3 val/test disjoint from replay train (row + project): PASS",
    }


if __name__ == "__main__":
    main()
