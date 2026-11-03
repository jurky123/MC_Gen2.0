"""Read legacy artifacts without modifying them; issue a new immutable catalog."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from .catalog import assign_splits, boolean, write_catalog, atomic_json
from .images import content_hash, native_grid


def migrate_pairs(project, output, size=16, seed=0, fractions=(0.9, 0.05, 0.05), include_text=False):
    root = Path(project).resolve()
    pairs = root / "pairs/stylizer_v4"
    df = pd.read_parquet(pairs / "metadata.parquet")
    base = root / "data/build/mc_text2image32_wl"
    prompts = pd.read_parquet(base / "stage3_prompts.parquet", columns=["prompt_0"])
    fallback = pd.read_parquet(base / "grounded_prompts.parquet", columns=["prompt_0"])
    n = len(df)
    target_path, ref_path = pairs / "target.uint8.mmap", pairs / "ref.uint8.mmap"
    if target_path.stat().st_size != n * 32 * 32 * 4 or ref_path.stat().st_size != n * 128 * 128 * 4:
        raise ValueError("unexpected legacy mmap layout")
    targets = np.memmap(target_path, dtype=np.uint8, mode="r", shape=(n, 32, 32, 4))
    records, rejected, hashes = [], Counter(), {}
    for i, row in enumerate(df.to_dict("records")):
        source_row = int(row["row"])
        try:
            target = native_grid(targets[i], size)
        except ValueError:
            rejected["not_exact_native_grid"] += 1
            continue
        prompt = prompts.iloc[source_row, 0]
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = fallback.iloc[source_row, 0]
        if not isinstance(prompt, str) or not prompt.strip():
            rejected["missing_canonical_prompt"] += 1
            continue
        lineage = row.get("project_id")
        if pd.isna(lineage) or not str(lineage).strip():
            rejected["missing_lineage"] += 1
            continue
        tileable = boolean(row["tileable"])
        kind = "item" if row["asset_type"] == "item" else "tile" if tileable is True else "other"
        digest = content_hash(target)
        if source_row in hashes and hashes[source_row] != digest:
            raise ValueError(f"generator variants disagree on target row {source_row}")
        hashes[source_row] = digest
        records.append({
            "id": f"legacy-v4:{i}", "lineage": "mc:" + str(lineage),
            "target_hash": digest, "prompt": prompt.strip(), "asset_type": kind,
            "tileable": tileable, "generator": str(row["generator"]),
            "source_row": source_row, "generator_license": str(row.get("generator_license", "")),
            "target": {"path": str(target_path), "shape": [n, 32, 32, 4], "index": i},
            "reference": {"path": str(ref_path), "shape": [n, 128, 128, 4], "index": i},
        })
    if include_text:
        meta = pd.read_parquet(base / "metadata.parquet")
        path = base / "images.uint8.mmap"
        count = len(meta)
        if path.stat().st_size != count * 32 * 32 * 4:
            raise ValueError("unexpected text dataset layout")
        images = np.memmap(path, dtype=np.uint8, mode="r", shape=(count, 32, 32, 4))
        # Pair rows already provide these text targets; no duplicate text-only
        # copies are added. All other sources participate in the SAME split pass.
        for i, row in enumerate(meta.to_dict("records")):
            if i in hashes:
                continue
            try:
                target = native_grid(images[i], size)
            except ValueError:
                rejected["text_not_exact_native_grid"] += 1
                continue
            lineage = row.get("project_id")
            prompt = fallback.iloc[i, 0]
            if pd.isna(lineage) or not str(lineage).strip() or not isinstance(prompt, str) or not prompt.strip():
                rejected["text_missing_lineage_or_prompt"] += 1
                continue
            tileable = boolean(row["tileable"])
            kind = "item" if row["type"] == "item" else "tile" if tileable is True else "other"
            records.append({"id": f"legacy-text:{i}", "lineage": "mc:" + str(lineage),
                            "target_hash": content_hash(target), "prompt": prompt.strip(),
                            "asset_type": kind, "tileable": tileable, "generator": "none",
                            "source_row": i, "source_license": str(row.get("license", "")),
                            "target": {"path": str(path), "shape": [count, 32, 32, 4], "index": i},
                            "reference": None})
    if not records:
        raise ValueError("no eligible samples")
    assign_splits(records, seed, fractions)
    return write_catalog(output, records, {
        "native_size": size, "seed": seed, "fractions": list(fractions),
        "source": str(pairs), "prompt_source": str(base / "stage3_prompts.parquet"),
        "rejected": dict(rejected), "include_text": include_text, "legacy_weights_compatible": False,
        "note": "All tasks inherit these new splits; no legacy replay is implicitly mixed in.",
    })


def legacy_audit(project, output):
    root = Path(project)
    p = root / "pairs/stylizer_v4"
    df = pd.read_parquet(p / "metadata.parquet")
    split = json.loads((p / "splits.json").read_text())
    holdout = set(df.iloc[split["val"]]["row"].astype(int))
    result = {"unique_val_targets": len(holdout), "legacy_reports_valid_for_new_benchmark": False,
              "tileable_values": df.tileable.value_counts().to_dict(),
              "legacy_bool_true": int(df.tileable.map(bool).sum()), "overlaps": {}}
    for name in ("stage3_splits", "replay_splits"):
        roles = json.loads((root / f"data/build/mc_text2image32_wl/{name}.json").read_text())
        result["overlaps"][name] = len(holdout & set(roles["train"]))
    atomic_json(output, result)
    return result
