"""Regression coverage for cross-source leakage, image contracts and native flow."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcgen.catalog import assign_splits, audit, boolean, read_catalog
from mcgen.config import Experiment
from mcgen.dataset import CatalogDataset
from mcgen.evaluation import metrics, select_indices
from mcgen.images import composite, content_hash, decode, encode, native_grid, reference
from mcgen.model import PixelFlow
from mcgen.sampling import sample
from mcgen.smoke import fixture, tiny_config


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_strict_config_rejects_typo_and_bad_dims():
    cfg = tiny_config().to_dict()
    cfg["train"]["toroidal"] = True
    with pytest.raises(ValueError, match="unknown"):
        Experiment.from_dict(cfg)
    cfg = tiny_config().to_dict()
    cfg["model"]["width"] = 31
    with pytest.raises(ValueError):
        Experiment.from_dict(cfg)


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("false", False), ("true", True), ("unknown", None), (None, None)])
def test_boolean(value, expected):
    assert boolean(value) is expected


def test_boolean_rejects_ambiguous():
    with pytest.raises(ValueError):
        boolean("False-ish")


def test_rgba_roundtrip_and_transparent_rgb_invariance():
    a = np.zeros((8, 8, 4), np.uint8)
    a[0, 0] = [100, 60, 20, 128]
    a[1, 1] = [141, 57, 180, 0]
    b = decode(encode(a))
    assert np.array_equal(b[0, 0], a[0, 0])
    assert np.array_equal(b[1, 1], [0, 0, 0, 0])
    a[1, 1, :3] = 255
    assert content_hash(a) == content_hash(b)
    assert np.array_equal(composite(a), composite(b))


def test_native_grid_requires_exact_cells():
    a = np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4)
    a[..., 3] = 255
    big = a.repeat(2, 0).repeat(2, 1)
    assert np.array_equal(native_grid(big, 8), a)
    big[1, 1, 0] += 1
    with pytest.raises(ValueError, match="exact"):
        native_grid(big, 8)


def test_reference_preserves_aspect_ratio_and_alpha():
    a = np.zeros((8, 16, 4), np.uint8)
    a[..., :] = [255, 0, 0, 255]
    im = decode(reference(a, 16))
    assert np.all(im[:4, :, 3] == 0)
    assert np.all(im[4:12, :, 3] == 255)
    assert np.all(im[4:12, :, 1] == 0)


def row(i, project, content):
    return {"id": str(i), "lineage": project, "target_hash": content,
            "tileable": False, "asset_type": "item", "prompt": "item"}


def test_transitive_project_content_split_and_order_independence():
    rows = [row(0, "a", "x"), row(1, "b", "x"), row(2, "b", "y"), row(3, "c", "y")]
    assign_splits(rows)
    assert len({r["split_group"] for r in rows}) == 1
    reordered = copy.deepcopy(list(reversed(rows)))
    assign_splits(reordered)
    assert {r["id"]: r["split"] for r in rows} == {r["id"]: r["split"] for r in reordered}
    rows[-1]["split"] = "test" if rows[0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="leaks"):
        audit(rows)


def test_catalog_detects_source_and_manifest_mutation(tmp_path):
    path = fixture(tmp_path)
    records, _ = read_catalog(path)
    target = Path(records[0]["target"]["path"])
    with target.open("r+b") as f:
        f.write(b"\x03")
    with pytest.raises(ValueError, match="content changed"):
        read_catalog(path)
    (path / "samples.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest changed"):
        read_catalog(path)


def test_text_and_reference_share_holdout(tmp_path):
    path = fixture(tmp_path)
    rows, _ = read_catalog(path)
    text = CatalogDataset(rows, 8, 16, "train", task="text")
    ref = CatalogDataset(rows, 8, 16, "val", task="reference")
    assert not {r["target_hash"] for r in text.records} & {r["target_hash"] for r in ref.records}
    assert not text[0]["tileable"]
    assert not text.groups[True]  # pure text sampling is uniform over targets


def test_remote_text_revision_cannot_float():
    cfg = tiny_config().to_dict()
    cfg["text"] = {"backend": "hf", "model": "example/model", "revision": "main"}
    with pytest.raises(ValueError, match="commit SHA"):
        Experiment.from_dict(cfg)


def test_metrics_transparency_empty_alpha_and_correct_units():
    a = np.zeros((8, 8, 4), np.uint8)
    b = a.copy()
    b[..., :3] = 200
    score = metrics(a, b)
    assert score["composite_mae"] == 0
    assert score["alpha_iou"] == 1
    a[..., 3] = b[..., 3] = 255
    assert metrics(a, b)["composite_mae"] == 200
    assert metrics(a, b)["edge_mae"] == 0


def test_balanced_eval_never_repeats_target():
    rows = [{**row(i, str(i), str(i // 2)), "generator": "a" if i % 2 else "b"} for i in range(20)]
    picked = select_indices(rows, 20, 0)
    assert len(picked) == len({rows[i]["target_hash"] for i in picked}) == 10
    assert picked == select_indices(rows, 20, 0)


def opened_model():
    torch.manual_seed(0)
    m = PixelFlow(tiny_config().model).eval()
    with torch.no_grad():
        m.output.weight.normal_(std=0.1)
        for block in m.blocks:
            block.modulation[-1].bias.fill_(0.2)
    return m


def test_null_reference_and_padding_do_not_leak():
    m = opened_model()
    ids = torch.ones(2, 5, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    x, t = torch.randn(2, 4, 8, 8), torch.ones(2) * 0.5
    ref = torch.randn(2, 4, 16, 16)
    absent = torch.zeros(2, dtype=torch.bool)
    a = m(x, t, m.context(ids, mask, None, None))
    b = m(x, t, m.context(ids, mask, ref, absent))
    assert torch.allclose(a, b, atol=1e-6)
    c = m(x, t, m.context(ids, mask, ref, ~absent))
    assert not torch.allclose(a, c, atol=1e-5)
    null = torch.zeros_like(mask)
    assert torch.allclose(m(x, t, m.context(ids, null, None, None)),
                          m(x, t, m.context(ids * 200, null, None, None)), atol=1e-6)


def test_per_sample_noise_reproducible_and_zero_cfg_unconditional():
    m = opened_model()
    ids = torch.ones(2, 5, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    a = sample(m, ids, mask, seeds=[1, 2], steps=2, text_scale=0, ref_scale=0)
    b = sample(m, ids * 3, mask, seeds=[1, 2], steps=2, text_scale=0, ref_scale=0)
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])
    for i in range(2):
        single = sample(m, ids[i:i+1], mask[i:i+1], seeds=[i + 1], steps=2, text_scale=0, ref_scale=0)
        assert torch.allclose(a[i], single[0], atol=1e-5)


def test_reference_path_has_gradients():
    m = opened_model().train()
    ids = torch.ones(2, 5, dtype=torch.long)
    context = m.context(ids, ids.bool(), torch.randn(2, 4, 16, 16), None)
    m(torch.randn(2, 4, 8, 8), torch.ones(2) * 0.3, context).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in m.reference_encoder.parameters()) > 0


def test_image_import_and_global_merge(tmp_path):
    from PIL import Image
    from mcgen.ingest import import_images, merge_catalogs
    dirs = []
    for j in range(2):
        source = tmp_path / str(j)
        source.mkdir()
        a = np.zeros((8, 8, 4), np.uint8)
        a[2:6, 2:6] = [10, 20, 30, 255]
        Image.fromarray(a).save(source / "item.png")
        manifest = source / "input.jsonl"
        manifest.write_text(json.dumps({"id": f"sample-{j}", "lineage": f"project-{j}",
                             "prompt": "item", "asset_type": "item", "tileable": "false",
                             "target": "item.png", "reference": "item.png"}) + "\n")
        import_images(manifest, source / "built", size=8, ref_size=16, seed=j)
        dirs.append(source / "built/catalog")
    merge_catalogs(dirs, tmp_path / "merged")
    records, _ = read_catalog(tmp_path / "merged")
    assert len(records) == 2
    assert len({r["split"] for r in records}) == 1
    assert all(r["tileable"] is False for r in records)
