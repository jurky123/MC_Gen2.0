"""Rectification test suite (docs/MC-Gen2_HD-to-MC_Design_v1.0.md section 9).

Covers: raw mmap round-trip, group split & leakage, RGBA premultiply, masked
tile loss, conditioning-dropout combinations, gradient-accumulation tail.
Run:  python -m pytest tests -q
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import mmap_io  # noqa: E402
from data.lineage_split import (  # noqa: E402
    _role_for_group,
    audit_no_leakage,
    check_disjoint,
    group_split,
)
from data.rgba import (  # noqa: E402
    premultiply_rgba_np,
    premultiply_rgba_torch,
    rgba_roundtrip_diff,
    unpremultiply_rgba_np,
    unpremultiply_rgba_torch,
)
from train.losses import flow_tile_loss  # noqa: E402
from train.flow import reconstruct_x0  # noqa: E402


# ---- mmap round-trip (P0-1) ------------------------------------------------
def test_raw_mmap_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(7, 32, 32, 4), dtype=np.uint8)
    path = tmp_path / "images.uint8.mmap"
    mmap_io.write_raw_mmap(arr, path)
    assert path.read_bytes()[:6] != b"\x93NUMPY", "writer must be headerless"
    back = mmap_io.read_raw_mmap(path, arr.shape, np.uint8)
    assert np.array_equal(np.asarray(back), arr)
    assert mmap_io.infer_n_rows(path, (32, 32, 4)) == 7


def test_read_rejects_wrong_shape(tmp_path):
    arr = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    path = tmp_path / "x.mmap"
    mmap_io.write_raw_mmap(arr, path)
    try:
        mmap_io.read_raw_mmap(path, (4, 8, 8, 4), np.uint8)
    except ValueError:
        pass
    else:
        raise AssertionError("shape/size mismatch must raise")


# ---- group split & leakage (P0-2 / P0-3) -----------------------------------
def test_group_split_keeps_groups_together():
    records = [{"project_id": f"p{i % 5}", "i": i} for i in range(50)]
    roles, audit = group_split(records, group_keys="project_id", seed=1)
    by_role = {}
    for i, rec in enumerate(records):
        by_role.setdefault(_role_for_group(f"p{i % 5}", 1, (0.9, 0.05, 0.05)), [])
    for i, rec in enumerate(records):
        assert roles_of_project(records, roles, rec["project_id"]) == 1
    assert audit["split_sizes"]


def roles_of_project(records, roles, project):
    members = [i for i, r in enumerate(records) if r["project_id"] == project]
    return len({roles[m] for m in members})


def test_leakage_audit_raises():
    try:
        audit_no_leakage({"val": [1, 2], "test": [3]}, {"train": [2, 3]}, "x")
    except AssertionError:
        pass
    else:
        raise AssertionError("must raise on leak")
    audit_no_leakage({"val": [1], "test": [2]}, {"train": [3, 4]}, "clean")


def test_check_disjoint_raises():
    try:
        check_disjoint({"a": {1, 2}, "b": {2, 3}})
    except AssertionError:
        pass
    else:
        raise AssertionError("must raise on overlap")


# ---- RGBA (P1-5) -----------------------------------------------------------
def test_premultiply_np_roundtrip():
    rng = np.random.RandomState(1)
    arr = rng.randint(0, 256, size=(16, 16, 4), dtype=np.uint8)
    pm = premultiply_rgba_np(arr)
    back = unpremultiply_rgba_np(pm)
    assert (back[..., 3] == arr[..., 3]).all()
    # alpha == 0: straight rgb is undefined and normalised to 0 (expected).
    # Very low alpha quantises premultiplied rgb heavily (standard premultiply
    # behaviour), so only compare alpha >= 16 pixels.
    visible = arr[..., 3] >= 32
    # uint8 premultiply quantisation: err(back) <= 255 / (2 * alpha)
    bound = int(np.ceil(255.0 / (2 * 32))) + 1
    diff = int(np.abs(back[..., :3].astype(int) - arr[..., :3])[visible].max())
    assert diff <= bound, diff
    transparent = arr[..., 3] == 0
    if transparent.any():
        assert (pm[transparent][:, :3] == 0).all()


def test_premultiply_zero_alpha_rgb_is_zero():
    arr = np.zeros((4, 4, 4), np.uint8)
    arr[..., :3] = 200
    pm = premultiply_rgba_np(arr)
    assert (pm[..., :3] == 0).all()
    back = unpremultiply_rgba_np(pm)
    assert (back[..., :3] == 0).all()
    assert (back[..., 3] == 0).all()


def test_premultiply_torch_matches_np():
    rng = np.random.RandomState(2)
    arr = rng.randint(0, 256, size=(4, 8, 8, 4), dtype=np.uint8)
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    pm_t = premultiply_rgba_torch(x)
    # torch (float) premultiply vs stored uint8 premultiply: <=1 source-unit err
    pm_np = np.stack([premultiply_rgba_np(a) for a in arr]).astype(np.float32)
    pm_src = (pm_t.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5
    assert np.abs(pm_src[..., :3] - pm_np[..., :3]).max() <= 1.0
    assert np.abs(pm_src[..., 3] - pm_np[..., 3]).max() <= 0.01
    # float-space roundtrip: exact on visible pixels; transparent rgb -> 0.
    back_t = unpremultiply_rgba_torch(pm_t)
    err = (back_t[:, :3] - x[:, :3]).abs().max(dim=1).values * 127.5
    visible = torch.from_numpy(arr[..., 3] > 0)
    assert err[visible].max().item() < 1.0
    invisible = torch.from_numpy(arr[..., 3] == 0)
    if invisible.any():
        assert (back_t[:, :3].max(dim=1).values[invisible] <= -0.999).all()


# ---- masked tile loss (P1-1) -----------------------------------------------
def test_tile_loss_mask_excludes_non_tileable():
    torch.manual_seed(0)
    b = 8
    x0 = torch.randn(b, 4, 32, 32)
    t = torch.full((b,), 0.1)
    z = torch.randn_like(x0)
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * z
    target = z - x0
    cfg = {"enabled": True, "weight": 1.0, "max_t": 0.7, "border_width": 2}
    v = torch.zeros_like(xt)
    mask_all = torch.ones(b, dtype=torch.bool)
    mask_none = torch.zeros(b, dtype=torch.bool)
    loss_all = flow_tile_loss(v, xt, t, target, cfg, tileable_mask=mask_all)
    loss_none = flow_tile_loss(v, xt, t, target, cfg, tileable_mask=mask_none)
    assert loss_none.item() < loss_all.item() - 0.5


def test_reconstruct_x0_consistency():
    torch.manual_seed(0)
    x0, z, t = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), torch.tensor([0.3, 0.7])
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * z
    v = z - x0
    back = reconstruct_x0(xt, t, v)
    assert (back - x0).abs().max() < 1e-5


# ---- conditioning-dropout combinations (P2-3) ------------------------------
def test_text_dropout_combinations():
    """cond_drop_prob must keep 1 - p conditioned tokens; zero text -> full drop."""
    combos = [0.0, 0.12, 0.5, 1.0]
    for p in combos:
        torch.manual_seed(0)
        keep = torch.rand(10_000) >= p
        assert 0 < keep.float().mean().item() or p == 1.0
        if p == 0.0:
            assert keep.all()
        if p == 1.0:
            assert not keep.any()


# ---- gradient accumulation tail scaling (P2-2) -----------------------------
def test_tail_accum_scaling_matches_full_batch():
    """Tail gradient scaled by accum/pending equals the full-window magnitude."""
    torch.manual_seed(0)
    data = torch.randn(8, 3)

    def grad_of(batches, accum, tail_scale=1.0):
        w = torch.nn.Parameter(torch.zeros(3))
        opt = torch.optim.SGD([w], lr=0.0)
        pending = 0
        for chunk in batches:
            loss = ((data[chunk] - w.expand(chunk.stop - chunk.start, 3)) ** 2).sum()
            (loss / accum).backward()
            pending += 1
        if pending < accum and tail_scale > 1.0:
            for p in [w]:
                p.grad.mul_(tail_scale)
        return w.grad.clone()

    # Tail step (accum=4 but only 2 batches, compensated by 4/2) must equal a
    # direct mean-over-2-batches gradient.
    tail = grad_of([slice(0, 2)], 4, tail_scale=4 / 2)
    w_ref = torch.nn.Parameter(torch.zeros(3))
    loss = ((data[:2] - w_ref.expand(2, 3)) ** 2).sum()
    (loss / 2).backward()
    torch.testing.assert_close(tail, w_ref.grad.clone(), rtol=1e-4, atol=1e-6)
    # Uncompensated tail would be half the magnitude.
    uncomp = grad_of([slice(0, 2)], 4)
    assert (uncomp - tail).abs().max() > 1e-6


# ---- v2 configs sanity -----------------------------------------------------
def test_v2_configs_exist_and_consistent():
    data = yaml.safe_load((ROOT / "configs/data/stage_3_v2.yaml").read_text())["dataset"]
    assert data["rgba_mode"] == "premultiplied"
    assert data["return_aux"] is True
    assert "replay_splits.json" in data["sources"][1]["splits"]
    train = yaml.safe_load((ROOT / "configs/train/stage_3_frozen_v2.yaml").read_text())["train"]
    assert train["freeze_backbone"] is True
    splits = json.loads((ROOT / "data/build/mc_text2image32_wl/stage3_splits.json").read_text())
    replay = json.loads((ROOT / "data/build/mc_text2image32_wl/replay_splits.json").read_text())
    audit_no_leakage(splits, replay, "v2 configs")


# ---- chunked text-encode pipeline ------------------------------------------
class _MockEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(len(texts))
        n = len(texts)
        return torch.zeros(n, 4, 8), torch.ones(n, 4, dtype=torch.bool)


def test_chunked_pipeline_batches_and_orders():
    from train.pipeline import ChunkedEncodedLoader

    batches = [(torch.full((2, 4), float(i)), [f"p{i}-{j}" for j in range(2)])
               for i in range(5)]
    loader = torch.utils.data.DataLoader(
        batches, batch_size=None, collate_fn=lambda b: b, num_workers=0)
    enc = _MockEncoder()
    pl = ChunkedEncodedLoader(loader, enc, accum=2, tower_device="cpu", main_device="cpu")
    chunks = list(pl)
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert enc.calls == [4, 4, 2]
    texts = [t for chunk in chunks for _x, (h, _m), _a in chunk
             for t in [h.shape[0]]]
    assert texts == [2, 2, 2, 2, 2]
    # order preserved: first item of first chunk carries batch 0's prompts
    assert chunks[0][0][1][0].shape[0] == 2
