"""Small local end-to-end fixture, explicitly not a trained production model."""
from pathlib import Path
import numpy as np
from PIL import Image

from .catalog import assign_splits, write_catalog
from .config import Experiment
from .images import content_hash


def tiny_config():
    return Experiment.from_dict({
        "model": {"size": 8, "width": 32, "depth": 2, "heads": 4,
                  "ref_size": 16, "text_dim": 32, "text_length": 48},
        "text": {"backend": "byte"},
        "train": {"steps": 6, "warmup": 1, "batch_size": 4, "eval_every": 3,
                  "val_samples": 8, "lr": 0.001,
                  "task_mix": {"text": .4, "reference": .25, "joint": .25, "unconditional": .1}}, "task": "joint"})


def fixture(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    targets, refs, records = [], [], []
    for i in range(40):
        a = np.zeros((8, 8, 4), dtype=np.uint8)
        a[1:7, 2:6] = [30 + i * 5, 100 + i, 80, 255]
        a[2, 2] = [255, 255, 255, 128]
        targets.append(a)
        refs.append(np.array(Image.fromarray(a).resize((16, 16), Image.Resampling.NEAREST)))
        records.append({"id": f"fixture:{i}", "lineage": f"fixture-project:{i}",
                        "target_hash": content_hash(a), "prompt": f"green bottle {i}",
                        "asset_type": "item", "tileable": False, "generator": "fixture",
                        "target": {"path": str(directory / "target.mmap"), "shape": [40, 8, 8, 4], "index": i},
                        "reference": {"path": str(directory / "reference.mmap"), "shape": [40, 16, 16, 4], "index": i}})
    np.stack(targets).tofile(directory / "target.mmap")
    np.stack(refs).tofile(directory / "reference.mmap")
    assign_splits(records, fractions=(0.6, 0.2, 0.2))
    write_catalog(directory / "catalog", records, {"native_size": 8, "fixture_only": True})
    return directory / "catalog"
