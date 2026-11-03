import copy
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcgen.catalog import read_catalog, audit
from mcgen.conditioning import TextEncoder
from mcgen.dataset import CatalogDataset
from mcgen.evaluation import audit_external
from mcgen.images import reference, decode, apply_alpha
from mcgen.prepare import project_holdout, prepare_catalog
from mcgen.smoke import fixture, tiny_config
from mcgen.tasks import TaskSampler
from mcgen.training import Trainer
from mcgen.runtime import Runtime


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def row(i, project, content):
    return dict(id=str(i), lineage=project, target_hash=content, prompt="bottle", asset_type="item", tileable=False)


def test_bridge_duplicates_quarantined_without_merging_projects():
    rows = [row(0,"a","common"),row(1,"b","common"),row(2,"a","a-only"),row(3,"b","b-only"),row(4,"c","c-only")]
    kept, rejected, plan, report = project_holdout(rows, assignments={"a":"train","b":"val","c":"test"})
    assert {r["id"] for r in rejected} == {"0","1"}
    assert len({r["split"] for r in kept}) == 3
    assert report["quarantined_unique_targets"] == 1
    audit(kept)
    assert "split" not in rows[0]
    again = project_holdout(list(reversed(rows)), assignments=plan)
    assert {r["id"]:r["split"] for r in kept} == {r["id"]:r["split"] for r in again[0]}


def test_compaction_preserves_content_and_source_catalog(tmp_path):
    source = fixture(tmp_path / "source")
    rows, old = read_catalog(source)
    plan = {r["lineage"]:r["split"] for r in rows}
    report = prepare_catalog(source, tmp_path / "new", assignments=plan)
    new, meta = read_catalog(tmp_path / "new/catalog")
    assert report["native_target_bytes"] == 40*8*8*4
    assert old == read_catalog(source)[1]
    assert {r["target_hash"] for r in new} == {r["target_hash"] for r in rows}
    assert all(r["alpha_mode"] == "continuous" for r in new)
    assert all("source_target" in r for r in new)


def test_task_sampling_preserves_requested_modes_without_dropout(tmp_path):
    path = fixture(tmp_path / "source")
    rows, _ = read_catalog(path)
    # A large duplicated reference pool must not multiply exposure in text mode.
    rows.extend([dict(r,id=r["id"]+":variant") for r in rows[:10]])
    ds = CatalogDataset(rows,8,16,"train")
    cfg=tiny_config()
    sampler=TaskSampler(ds,cfg)
    indices,modes=sampler.sample(20000,torch.Generator().manual_seed(9))
    for mode, probability in cfg.train.task_mix.items():
        assert abs(modes.count(mode)/len(modes)-probability)<.015
    assert len(sampler.pools["text"]) == len({r["target_hash"] for r in ds.records})
    assert all(ds.records[i].get("reference") for i,m in zip(indices,modes) if m in ("reference","joint"))
    for r in rows:
        r["reference"]=None
    with pytest.raises(ValueError,match="no training targets"):
        TaskSampler(CatalogDataset(rows,8,16,"train"),cfg)


def test_reference_alpha_distinguishes_white_subject_from_background():
    a=np.zeros((8,8,4),np.uint8); a[2:6,2:6]=255
    b=np.full_like(a,255)
    assert not torch.equal(reference(a,16), reference(b,16))
    decoded=decode(reference(a,8))
    assert np.array_equal(decoded,a)
    assert np.all(apply_alpha(a,"opaque")[...,3]==255)


def test_external_benchmark_audits_validation_leakage():
    training=[dict(row(0,"a","x"),split="val")]
    test=[dict(row(1,"b","x"),split="test")]
    with pytest.raises(ValueError,match="overlaps"):
        audit_external(training,test,"test")
    test[0]["target_hash"]="new"
    assert not any(audit_external(training,test,"test")["collisions"].values())


def test_bounded_cache_reorders_and_deduplicates_without_batch_storage():
    # Exercise real LRU logic with a deterministic frozen feature provider.
    enc=TextEncoder.__new__(TextEncoder)
    enc.cfg=type("Cfg",(),{"backend":"hf"})()
    enc.cache=OrderedDict();enc.cache_bytes=enc.hits=enc.misses=0
    enc.cache_limit=2*(4*3*4+4)
    calls=[]
    def encode(prompts,device):
        calls.append(prompts)
        return torch.stack([torch.full((4,3),float(ord(p))) for p in prompts]),torch.ones(len(prompts),4,dtype=torch.bool)
    enc._encode=encode
    first,_=enc.encode(["a","b","a"],"cpu")
    second,_=enc.encode(["b","a"],"cpu")
    assert calls==[["a","b"]]
    assert torch.equal(first[0],second[1])
    enc.encode(["c"],"cpu")
    assert enc.cache_bytes<=enc.cache_limit and len(enc.cache)==2
    assert all(v[0].untyped_storage().nbytes()==48 for v in enc.cache.values())


def test_reference_specialist_runs_and_rejects_missing_reference(tmp_path):
    cfg=tiny_config();cfg.task="reference"
    cfg.train.task_mix={"text":0.,"reference":.9,"joint":0.,"unconditional":.1}
    path=fixture(tmp_path/"fixture")
    checkpoint=Trainer(cfg,path,tmp_path/"train").run(stop_after=1)
    runtime=Runtime(checkpoint)
    with pytest.raises(ValueError,match="requires a reference"):
        runtime.generate(["bottle"],steps=1)
    out=runtime.generate([""],[np.full((8,8,4),255,np.uint8)],steps=1)
    assert out[0].shape==(8,8,4)
    assert (tmp_path/"train/best_reference_flow.pt").exists()
    assert not (tmp_path/"train/best_text_flow.pt").exists()


def test_matched_target_scope_keeps_text_and_reference_populations_equal(tmp_path):
    rows, _ = read_catalog(fixture(tmp_path / "source"))
    for r in rows[::2]:
        r["reference"] = None
    text = CatalogDataset(rows, 8, 16, "train", task="text", target_scope="paired")
    ref = CatalogDataset(rows, 8, 16, "train", task="reference", target_scope="paired")
    assert {r["target_hash"] for r in text.records} == {r["target_hash"] for r in ref.records}
    broad = CatalogDataset(rows, 8, 16, "train", task="text")
    assert len(broad) > len(text)


def test_paired_evaluation_uses_identical_reference_cases_for_every_task(tmp_path, monkeypatch):
    from mcgen.catalog import write_catalog
    from mcgen.evaluation import evaluate
    rows, meta = read_catalog(fixture(tmp_path / "source"))
    rows = [dict(r, id="caption:" + r["id"], reference=None, generator="none") for r in rows] + rows
    catalog = tmp_path / "catalog"
    meta = write_catalog(catalog, rows, {"native_size": 8})
    cfg = tiny_config(); cfg.target_scope = "paired"
    class FakeRuntime:
        config = cfg
        metadata = {"catalog_fingerprint": meta["fingerprint"]}
        def __init__(self, *args): pass
        def generate(self, *args, **kwargs):
            return [np.full((8, 8, 4), 255, np.uint8)]
    monkeypatch.setattr("mcgen.evaluation.Runtime", FakeRuntime)
    checkpoint = tmp_path / "checkpoint"; checkpoint.write_text("fixture")
    for task in ("joint", "text", "reference"):
        cfg.task = task
        report = evaluate(checkpoint, catalog, tmp_path / task, n=3, steps=1)
        if task == "joint":
            expected = report["selected_ids"]
            assert all(report["summary"][m]["n"] == 3 for m in ("text", "reference", "joint"))
        else:
            assert report["selected_ids"] == expected
