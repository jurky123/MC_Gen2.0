import csv
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcgen.benchmark import run_benchmark, summarize_review
from mcgen.catalog import file_hash


def test_blind_benchmark_tracks_both_tasks_and_pending_review(tmp_path, monkeypatch):
    class Runtime:
        config = type("Config", (), {"asset_type":"item", "task":"joint"})()
        metadata = {"task":"joint"}
        def __init__(self,*args): pass
        def generate(self,prompts,references=None,**kwargs):
            return [np.full((8,8,4),255,np.uint8)]
    monkeypatch.setattr("mcgen.benchmark.Runtime",Runtime)
    checkpoint=tmp_path/"checkpoint";checkpoint.write_text("fixture")
    ref=tmp_path/"ref.png";Image.fromarray(np.full((8,8,4),255,np.uint8)).save(ref)
    manifest=tmp_path/"cases.jsonl"
    manifest.write_text(json.dumps(dict(id="case",asset_type="item",prompt="white object",criteria="white foreground",
       reference="ref.png",reference_sha256=file_hash(ref),provenance="test fixture"))+"\n")
    out=tmp_path/"out"
    run_benchmark(checkpoint,manifest,out,seeds=(7,),steps=1)
    assert {r["mode"] for r in json.loads((out/"key.json").read_text())}=={"text","reference","joint"}
    assert summarize_review(out)["pending"]==3
    with (out/"review.csv").open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for r in rows:
        r.update(semantic_0_4="3",pixel_style_0_4="4",usable_0_1="1")
    with (out/"review.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report=summarize_review(out)
    assert report["complete"] and len(report["by_task"])==3
    rows.append(rows[0])
    with (out/"review.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="IDs"):
        summarize_review(out)
