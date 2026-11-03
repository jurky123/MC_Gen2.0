"""Versioned semantic cases and blinded review, without a single target-image proxy."""
import csv
import json
from pathlib import Path
import random
import numpy as np

from PIL import Image

from .catalog import atomic_json, file_hash
from .images import export
from .runtime import Runtime


def run_benchmark(checkpoint, manifest, output, *, device="cpu", text_device=None, steps=20, seeds=(0, 1, 2)):
    manifest, output = Path(manifest).resolve(), Path(output)
    if output.exists():
        raise ValueError("benchmark output exists")
    with manifest.open() as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("benchmark requires unique case IDs")
    runtime = Runtime(checkpoint, device, text_device)
    selected = [c for c in cases if runtime.config.asset_type in ("all", c["asset_type"])]
    jobs = []
    for case in selected:
        if not isinstance(case.get("criteria"), str) or not case["criteria"].strip():
            raise ValueError("each case needs explicit human acceptance criteria")
        ref = None
        if case.get("reference"):
            path = manifest.parent / case["reference"]
            if file_hash(path) != case.get("reference_sha256"):
                raise ValueError("reference SHA256 missing or changed")
            if not case.get("provenance"):
                raise ValueError("reference cases require provenance")
            with Image.open(path) as im:
                ref = im.convert("RGBA")
        modes = ([] if runtime.config.task == "reference" else ["text"])
        if ref is not None and runtime.config.task != "text":
            modes += ["reference"]
            if runtime.config.task == "joint":
                modes += ["joint"]
        for seed in seeds:
            for mode in modes:
                jobs.append((case, mode, seed, ref))
    if not jobs:
        raise ValueError("no benchmark cases apply to this checkpoint; reference specialists need reference cases")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    random.Random(0).shuffle(jobs)
    key, review = [], []
    for i, (case, mode, seed, ref) in enumerate(jobs):
        blind_id = f"sample_{i:05d}"
        prompt = "" if mode == "reference" else case["prompt"]
        a = runtime.generate([prompt], [ref] if mode != "text" else None, seeds=[seed], steps=steps)[0]
        export(a).save(output / "images" / f"{blind_id}.png")
        preview = np.tile(a, (3, 3, 1)) if case["asset_type"] == "tile" else a
        export(preview, 8).save(output / "images" / f"{blind_id}_preview.png")
        if ref is not None and mode != "text":
            ref.save(output / "images" / f"{blind_id}_reference.png")
        key.append({"blind_id": blind_id, "case_id": case["id"], "mode": mode, "seed": seed,
                    "asset_type": case["asset_type"], "provenance": case.get("provenance", "authored_text")})
        criterion = case.get("reference_criteria", "Preserve reference subject and silhouette; produce clean Minecraft pixel art.") if mode == "reference" else case["criteria"]
        review.append([blind_id, prompt, criterion, f"images/{blind_id}_preview.png",
                       f"images/{blind_id}_reference.png" if ref is not None and mode != "text" else "", "", "", "", ""])
        print(f"benchmark {i + 1}/{len(jobs)}", flush=True)
    atomic_json(output / "key.json", key)
    atomic_json(output / "benchmark.json", {"schema_version": 1, "checkpoint_sha256": file_hash(checkpoint),
                "manifest_sha256": file_hash(manifest), "model": runtime.metadata,
                "steps": steps, "seeds": list(seeds), "samples": len(jobs),
                "note": "Authored semantic/provenance review, not a claim of verified OOD. Keep key.json hidden during scoring."})
    with (output / "review.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["blind_id", "prompt", "criteria", "preview", "reference", "semantic_0_4", "pixel_style_0_4", "usable_0_1", "notes"])
        writer.writerows(review)
    return output


def summarize_review(directory):
    directory = Path(directory)
    key = {r["blind_id"]: r for r in json.loads((directory / "key.json").read_text())}
    with (directory / "review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["blind_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(key):
        raise ValueError("review IDs must match benchmark exactly")
    groups = {}
    pending = 0
    for row in rows:
        fields = ("semantic_0_4", "pixel_style_0_4", "usable_0_1")
        if any(not row[f].strip() for f in fields):
            pending += 1
            continue
        values = [int(row[f]) for f in fields]
        if not (0 <= values[0] <= 4 and 0 <= values[1] <= 4 and values[2] in (0, 1)):
            raise ValueError("review scores out of range")
        item = key[row["blind_id"]]
        group = item["asset_type"] + "/" + item["mode"]
        groups.setdefault(group, []).append(values)
    report = {"complete": pending == 0, "pending": pending, "total": len(rows),
              "by_task": {g: {"n": len(v), "semantic_mean": sum(x[0] for x in v)/len(v),
                              "pixel_style_mean": sum(x[1] for x in v)/len(v),
                              "usable_fraction": sum(x[2] for x in v)/len(v)} for g,v in groups.items()},
              "note": "No automatic promotion. Compare task-specific baselines and review failures."}
    atomic_json(directory / "review_summary.json", report)
    return report
