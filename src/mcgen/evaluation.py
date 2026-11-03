"""Transparent-aware native-grid metrics and reproducible dual-task benchmarks."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .catalog import atomic_json, file_hash, read_catalog
from .dataset import CatalogDataset
from .images import composite, decode, export, native_grid, rgba
from .runtime import Runtime


def metrics(predicted, target):
    p, t = rgba(predicted), rgba(target)
    if p.shape != t.shape:
        raise ValueError("metrics require identical native grids")
    pm, tm = p[..., 3] >= 128, t[..., 3] >= 128
    union = np.logical_or(pm, tm).sum()
    iou = float(np.logical_and(pm, tm).sum() / union) if union else 1.
    # Both white and black backgrounds expose alpha errors without scoring
    # meaningless transparent RGB. Error units are [0,255], not called L2.
    pw, tw = composite(p), composite(t)
    pb, tb = composite(p, 0), composite(t, 0)
    mae = float((np.abs(pw - tw).mean() + np.abs(pb - tb).mean()) / 2)
    edge = float((np.abs(np.diff(pw, axis=0) - np.diff(tw, axis=0)).mean() +
                  np.abs(np.diff(pw, axis=1) - np.diff(tw, axis=1)).mean()) / 2)
    visible = p[..., 3] > 8
    colors = np.unique(p[..., :3][visible], axis=0)
    target_colors = np.unique(t[..., :3][t[..., 3] > 8], axis=0)
    palette = None
    if len(colors) and len(target_colors):
        distance = np.abs(colors[:, None].astype(float) - target_colors[None].astype(float)).mean(-1)
        palette = float(distance.min(1).mean())
    return {"composite_mae": mae, "alpha_mae": float(np.abs(p[..., 3].astype(float) - t[..., 3]).mean()),
            "alpha_iou": iou, "edge_mae": edge, "visible_colors": int(len(colors)),
            "palette_mae": palette,
            "seam_mae": float((np.abs(pb[:, 0] - pb[:, -1]).mean() + np.abs(pb[0] - pb[-1]).mean()) / 2)}


def baseline(reference, size, colors=0):
    # This intentionally does not invent a foreground mask; an opaque input
    # remains opaque. Its limitation is explicit in the report.
    small = np.array(Image.fromarray(rgba(reference)).resize((size, size), Image.Resampling.BOX))
    return np.array(export(small, colors=colors))


def select_indices(records, n, seed):
    """Seeded round-robin strata; one row per target hash, no duplicate variants."""
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i in rng.permutation(len(records)):
        r = records[int(i)]
        buckets[(r["asset_type"], r.get("generator", "none"))].append(int(i))
    selected, seen = [], set()
    while buckets and len(selected) < n:
        for key in sorted(list(buckets)):
            bucket = buckets[key]
            while bucket and records[bucket[-1]]["target_hash"] in seen:
                bucket.pop()
            if not bucket:
                del buckets[key]
                continue
            i = bucket.pop()
            selected.append(i)
            seen.add(records[i]["target_hash"])
            if len(selected) == n:
                break
    return selected


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["mode"]].append(r)
    report = {}
    for mode, items in grouped.items():
        summary = {"n": len(items), "lineages": len({r["lineage"] for r in items})}
        for metric in items[0]["metrics"]:
            values = [r["metrics"][metric] for r in items if r["metrics"][metric] is not None]
            if not values:
                summary[metric] = None
                continue
            # Bootstrap project means, not correlated generator variants.
            by_project = defaultdict(list)
            for r in items:
                if r["metrics"][metric] is not None:
                    by_project[r["lineage"]].append(r["metrics"][metric])
            means = np.array([np.mean(v) for v in by_project.values()])
            rng = np.random.default_rng(0)
            ci = np.quantile(means[rng.integers(len(means), size=(1000, len(means)))].mean(1), [0.025, 0.975])
            summary[metric] = {"sample_mean": float(np.mean(values)), "project_mean": float(means.mean()),
                               "project_bootstrap_95": ci.tolist()}
        report[mode] = summary
    return report


def evaluate(checkpoint, catalog, output, *, n=64, seed=0, split="test", device="cpu",
             text_device=None, steps=20, training_catalog=None, target_scope=None):
    if n < 1 or split not in ("val", "test"):
        raise ValueError("evaluation requires n>0 and val/test split")
    out = Path(output)
    if out.exists():
        raise ValueError("evaluation output already exists; preserve prior reports")
    records, meta = read_catalog(catalog)
    runtime = Runtime(checkpoint, device, text_device)
    external = meta["fingerprint"] != runtime.metadata["catalog_fingerprint"]
    overlap_audit = None
    if external:
        if training_catalog is None:
            raise ValueError("external benchmark requires --training-catalog for overlap audit")
        training_rows, training_meta = read_catalog(training_catalog)
        if training_meta["fingerprint"] != runtime.metadata["catalog_fingerprint"]:
            raise ValueError("training catalog does not match checkpoint")
        overlap_audit = audit_external(training_rows, records, split)
    if meta["native_size"] != runtime.config.model.size:
        raise ValueError("benchmark native grid differs from checkpoint")
    cfg = runtime.config
    scope = target_scope or cfg.target_scope
    if scope not in ("all", "paired"):
        raise ValueError("target_scope must be all or paired")
    ds = CatalogDataset(records, cfg.model.size, cfg.model.ref_size, split, cfg.asset_type, "joint" if scope == "paired" else cfg.task, cfg.alpha_mode, cfg.style, scope)
    if scope == "paired":
        # Training may retain unpaired captions for a paired target. Benchmark
        # cases must select an actual reference-bearing row for every task,
        # otherwise one text case silently disappears from reference scoring.
        ds.records = [r for r in ds.records if r.get("reference")]
    indices = select_indices(ds.records, n, seed)
    out.mkdir(parents=True)
    (out / "images").mkdir()
    rows, thumbnails = [], []
    for k, index in enumerate(indices):
        r = ds.records[index]
        target = native_grid(ds.array(r["target"]), cfg.model.size)
        ref = ds.array(r["reference"]) if r.get("reference") else None
        outputs = {}
        if cfg.task != "reference":
            outputs["text"] = runtime.generate([r["prompt"]], seeds=[seed + k], steps=steps)[0]
        if ref is not None and cfg.task != "text":
            outputs["reference"] = runtime.generate([""], [ref], seeds=[seed + k], steps=steps, text_scale=0)[0]
            if cfg.task == "joint":
                outputs["joint"] = runtime.generate([r["prompt"]], [ref], seeds=[seed + k], steps=steps)[0]
            candidates = [i for i, v in enumerate(ds.records) if v["target_hash"] != r["target_hash"] and v.get("reference")]
            if candidates and cfg.task == "joint":
                j = candidates[(seed + k) % len(candidates)]
                wrong = ds.array(ds.records[j]["reference"])
                outputs["shuffled_reference"] = runtime.generate([r["prompt"]], [wrong], seeds=[seed + k], steps=steps)[0]
            outputs["box_resize"] = baseline(ref, cfg.model.size)
            outputs["box_palette16"] = baseline(ref, cfg.model.size, 16)
        Image.fromarray(target).save(out / "images" / f"{k:04d}_target.png")
        if ref is not None:
            Image.fromarray(ref).save(out / "images" / f"{k:04d}_reference.png")
        for mode, image in outputs.items():
            path = f"images/{k:04d}_{mode}.png"
            Image.fromarray(image).save(out / path)
            score = metrics(image, target)
            # Seam is only meaningful for explicitly tileable textures.
            if r["tileable"] is not True:
                score["seam_mae"] = None
            rows.append({"id": r["id"], "target_hash": r["target_hash"], "lineage": r["lineage"],
                         "generator": r.get("generator"), "asset_type": r["asset_type"], "prompt": r["prompt"],
                         "seed": seed + k, "mode": mode, "image": path, "metrics": score})
        thumbnails.append((r["prompt"], target, outputs))
        print(f"evaluated {k + 1}/{len(indices)}", flush=True)
    report = {"schema_version": 2, "checkpoint_sha256": file_hash(checkpoint),
              "catalog_fingerprint": meta["fingerprint"], "config": cfg.to_dict(),
              "external": external, "overlap_audit": overlap_audit, "target_scope": scope,
              "split": split, "seed": seed, "steps": steps, "solver": "heun",
              "text_scale": 2.0, "ref_scale": 2.0, "requested_n": n,
              "selected_ids": [ds.records[i]["id"] for i in indices], "summary": summarize(rows),
              "by_asset_type": {a: summarize([r for r in rows if r["asset_type"] == a]) for a in sorted({r["asset_type"] for r in rows})},
              "by_generator": {str(g): summarize([r for r in rows if r["generator"] == g]) for g in sorted({r["generator"] for r in rows}, key=str)},
              "notes": ["Pixel reconstruction metrics do not measure text semantics.",
                        "Review text fidelity and pixel-art quality in human_review.csv.",
                        "Palette baseline preserves reference alpha and does not extract a foreground.",
                        "Raw predictions are evaluated before export palette quantization."]}
    atomic_json(out / "report.json", report)
    with (out / "samples.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "human_review.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "mode", "image", "semantic_0_4", "pixel_style_0_4", "usable", "notes"])
        for row in rows:
            writer.writerow([row["id"], row["prompt"], row["mode"], row["image"], "", "", "", ""])
    # Paginated previews avoid unreadable 14,000-pixel sheets.
    modes = list(dict.fromkeys(m for _, _, outputs in thumbnails for m in outputs))
    for page in range(0, len(thumbnails), 8):
        group = thumbnails[page:page + 8]
        sheet = Image.new("RGB", ((len(modes) + 1) * 136, len(group) * 156 + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for j, name in enumerate(["target", *modes]):
            draw.text((j * 136, 2), name, fill="black")
        for i, (prompt, target, outputs) in enumerate(group):
            draw.text((0, 24 + i * 156), prompt[:100].encode("ascii", "replace").decode(), fill="black")
            for j, a in enumerate([target, *[outputs.get(m) for m in modes]]):
                if a is None:
                    continue
                im = Image.fromarray(composite(a).astype(np.uint8)).resize((128, 128), Image.Resampling.NEAREST)
                sheet.paste(im, (j * 136, 44 + i * 156))
        sheet.save(out / f"review_{page // 8:03d}.png")
    return report


def audit_external(training_rows, benchmark_rows, split):
    # Validation has influenced checkpoint selection, so exclude both train and
    # validation provenance from external claims, not just optimizer examples.
    exposed = [r for r in training_rows if r["split"] in ("train", "val")]
    selected = [r for r in benchmark_rows if r["split"] == split]
    collisions = {}
    for key in ("id", "lineage", "target_hash"):
        values = {r[key] for r in exposed}
        overlap = {r[key] for r in selected} & values
        collisions[key] = len(overlap)
    if any(collisions.values()):
        raise ValueError(f"external benchmark overlaps training/validation: {collisions}")
    # A reused synthetic reference can leak even when its paired target differs.
    def refs(rows):
        return {(r["reference"]["path"], r["reference"]["index"]) for r in rows if r.get("reference")}
    collisions["reference_source_rows"] = len(refs(exposed) & refs(selected))
    if collisions["reference_source_rows"]:
        raise ValueError("external benchmark reuses exposed reference rows")
    return {"selected_records": len(selected), "collisions": collisions,
            "limitation": "Reference identity uses source rows; unknown source ancestry and perceptual duplicates need human provenance review."}
