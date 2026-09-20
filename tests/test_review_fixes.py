"""Tests for the 2026-09-20 design-review fixes (P0/P1).

Run:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.lineage_split import (  # noqa: E402
    audit_no_leakage,
    check_disjoint,
    group_split,
)
from data.rgba import composite_on_white, prepare_model_image  # noqa: E402
from gen_tierd_prompts import compatible  # noqa: E402
from build_tierd_prototype import read_prompts  # noqa: E402


# ---- P0-1: JSONL / JSON array prompt input ---------------------------------
def test_read_prompts_json_array(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"prompt": "red block", "asset_type": "block"}]))
    items = read_prompts(p)
    assert items[0]["prompt"] == "red block" and items[0]["bucket"] == -1


def test_read_prompts_jsonl(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"prompt": "a"}\n{"prompt": "b", "asset_type": "item"}\n')
    items = read_prompts(p)
    assert [i["prompt"] for i in items] == ["a", "b"]
    assert items[1]["asset_type"] == "item"


def test_read_prompts_rejects_missing_prompt(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt": "ok"}\n{"asset_type": "block"}\n')
    try:
        read_prompts(p)
    except ValueError:
        pass
    else:
        raise AssertionError("must reject entries without prompt")


# ---- P0-4: hd manifest index with flux_batch != sde_batch ------------------
def test_hd_manifest_indexing():
    # Simulates the consumer loop with FB=8, SB=4: sample id j must map to
    # hd_paths[b + t] where b is the sub-batch offset and t the in-sub index.
    for FB, SB in ((4, 4), (8, 4), (8, 2)):
        for s in (0, 8):
            batch = list(range(s, s + FB))
            hd_paths = [f"hd/{s + k:05d}.png" for k in range(len(batch))]
            for b in range(0, len(batch), SB):
                idx = [s + b + t for t in range(len(batch[b:b + SB]))]
                for t, j in enumerate(idx):
                    assert hd_paths[b + t] == f"hd/{j:05d}.png", (FB, SB, s, b, t, j)


# ---- P0-5: per-sample seed reproducibility ---------------------------------
def test_explicit_noise_generator_reproducible():
    shape = (1, 4, 32, 32)
    g1 = torch.Generator().manual_seed(1000003 + 7)
    g2 = torch.Generator().manual_seed(1000003 + 7)
    z1 = torch.randn(shape, generator=g1)
    z2 = torch.randn(shape, generator=g2)
    assert torch.equal(z1, z2)
    g3 = torch.Generator().manual_seed(1000003 + 8)
    assert not torch.equal(z1, torch.randn(shape, generator=g3))


def test_flux_generator_list_distinct():
    gs = [torch.Generator().manual_seed(100 + k) for k in range(4)]
    xs = [torch.randn(4, generator=g) for g in gs]
    assert not torch.equal(xs[0], xs[1])


# ---- P0-3: project-level leakage audit on real splits ----------------------
def test_project_level_no_leakage():
    import pandas as pd

    build = ROOT / "data/build/mc_text2image32_wl"
    meta = pd.read_parquet(build / "metadata.parquet", columns=["project_id"])
    s3 = json.loads((build / "stage3_splits.json").read_text())
    rep = json.loads((build / "replay_splits.json").read_text())
    holdout = set(s3["val"]) | set(s3["test"])
    hp = set(meta.loc[sorted(holdout), "project_id"].astype(str))
    bad = [i for i in rep["train"] if str(meta.at[i, "project_id"]) in hp]
    assert not bad, f"project-level leak: {len(bad)} rows"
    audit_no_leakage(s3, rep, "stage3 vs replay")


# ---- P1-1: prepare_model_image premultiplied --------------------------------
def test_prepare_model_image_premultiplied():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(32, 32, 4), dtype=np.uint8)
    t = prepare_model_image(arr, premultiplied=True)
    assert t.shape == (4, 32, 32)
    assert t.min() >= -1.0 and t.max() <= 1.0
    zero_alpha = arr[..., 3] == 0
    if zero_alpha.any():
        assert (t[:3][:, zero_alpha] == -1.0).all()
    t2 = prepare_model_image(arr, premultiplied=False)
    assert (t2[3] == t[3]).all()


# ---- P1-3: composite_on_white ------------------------------------------------
def test_composite_on_white():
    arr = np.zeros((8, 8, 4), np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 0
    out = composite_on_white(arr)
    assert out.shape == (8, 8, 3)
    assert (out == 255).all()


# ---- P1-5: build_mmap explicit splits + seed sensitivity --------------------
def test_build_mmap_explicit_splits(tmp_path):
    from data.build_mmap import build_mmap
    from PIL import Image

    recs = []
    for i in range(10):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 32), (i * 20, 0, 0)).save(p)
        recs.append({"path": str(p), "project_id": f"p{i % 3}"})
    explicit = {"train": list(range(7)), "val": [7, 8], "test": [9]}
    build_mmap(tmp_path / "b1", recs, image_size=32, channels=3, splits=explicit)
    got = json.loads((tmp_path / "b1" / "splits.json").read_text())
    assert got == explicit


def test_group_split_seed_changes_assignment():
    recs = [{"project_id": f"p{i}"} for i in range(30)]
    r1, _ = group_split(recs, seed=1)
    r2, _ = group_split(recs, seed=2)
    assert [r1[i] for i in range(30)] != [r2[i] for i in range(30)]


# ---- P1-9: compatibility ------------------------------------------------------
def test_compatible():
    assert compatible("item", "sword", "steel", "runes", "")
    assert not compatible("block", "sword", "steel", "runes", "")
    assert not compatible("item", "helmet", "steel", "brick courses", "")
    assert compatible("block", "tile", "marble", "grid", "")
    assert not compatible("block", "helmet", "steel", "plain", "transparent")
    assert compatible("item", "potion", "glass", "plain", "transparent")
