import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import load_yaml
from data.build_mmap import MmapImageTextDataset


class LabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LabeledDataset(torch.utils.data.Dataset):
    """Wrap MmapImageTextDataset and return (x, label_id) per sample.

    Labels are resolved through ``base.index[i]`` (the mmap row), so shuffled
    DataLoader order cannot desynchronise images and labels (P2-1). The model
    input channel count follows the dataset's configured channels instead of a
    hardcoded 3.
    """

    def __init__(self, base, y):
        self.base = base
        self.y = y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.y[i])


class LabelClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_labeled_dataset(data_cfg_path, label_col, split="train"):
    data = load_yaml(data_cfg_path).get("dataset", {})
    ds = MmapImageTextDataset(
        images=data["images"],
        metadata=data["metadata"],
        splits=data["splits"],
        split=split,
        image_size=int(data.get("image_size", 32)),
        channels=int(data.get("channels", 3)),
        toroidal=False,
        text_mmap=data.get("text_mmap", ""),
    )
    labels = [ds.df.iloc[i][label_col] for i in ds.index]
    classes = sorted(set(labels))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([label_to_id[l] for l in labels], dtype=torch.long)
    return ds, y, classes


def train_classifier(data_cfg_path, label_col, epochs=10, lr=1e-3, out="checkpoints/classifier.pt"):
    ds, y, classes = load_labeled_dataset(data_cfg_path, label_col)
    labeled = LabeledDataset(ds, y)
    model = LabelClassifier(len(classes), in_channels=ds.channels)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(labeled, batch_size=64, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, target in loader:
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        print(f"epoch {ep} loss {tot / max(n, 1):.4f}")
    model.eval()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": classes}, out)
    print(f"classifier saved: {out}")
    return model, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data/stage_c.yaml")
    ap.add_argument("--label", default="material")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/classifier.pt")
    args = ap.parse_args()
    train_classifier(args.data, args.label, epochs=args.epochs, out=args.out)


if __name__ == "__main__":
    main()