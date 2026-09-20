"""Standalone validation + checkpoint promotion (recovery tool).

Computes the flow-MSE on a dataset config's val split for a finished
checkpoint and, when requested, promotes it to best.pt. Used when a run
completed its steps but the in-loop val/selection crashed.

    python scripts/val_select.py --ckpt checkpoints/stage_3_frozen_v2/latest.pt \
        --data configs/data/stage_3_v2.yaml --promote
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig, load_yaml  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from train.flow import rand_timesteps, sample_data_noise  # noqa: E402
from train.train import _collate, build_dataset  # noqa: E402


def validate(ckpt, data_cfg_path, device, batch=64, seed=0):
    sd = torch.load(ckpt, map_location="cpu")
    mcfg = ModelConfig.from_dict(sd["model_cfg"])
    model = MCFlowDiT(mcfg)
    # NOTE: validate the RAW model weights, never the EMA shadow. The EMA
    # average mixes randomly re-initialised conditioning weights (text_proj /
    # text_null) with trained ones and is functionally broken in this setup
    # (ema mse ~1.4 vs raw ~0.06 on the same val). All consumers (val_validate,
    # sample.py, --init-from) use the raw model.
    model.load_state_dict(sd["model"], strict=True)
    model.to(device).eval()
    tower_cfg = (sd.get("train_cfg") or {}).get("text_tower", {})
    enc = get_text_encoder(
        tower_cfg.get("model_name", "") or "/home/iflab/models/Qwen3-8B",
        device=tower_cfg.get("device") or device,
        dtype=tower_cfg.get("dtype", "bfloat16"),
        max_length=int(tower_cfg.get("max_length", mcfg.max_text_tokens)),
        revision=tower_cfg.get("revision") or None,
        instruction=tower_cfg.get("instruction", ""),
        layers=tower_cfg.get("layers"),
        pad_bucket=int(tower_cfg.get("pad_bucket", 0)),
    )
    data = load_yaml(data_cfg_path)
    val_ds = build_dataset(data_cfg_path, "val", None)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False,
                                         collate_fn=_collate)
    torch.manual_seed(seed)
    total = n = 0
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            x, text = b[0].to(device), b[1]
            if isinstance(text, (list, tuple)):
                text, text_mask = enc.encode(list(text))
                text, text_mask = text.to(device), text_mask.to(device)
            else:
                text, text_mask = text.to(device), None
            bt = x.shape[0]
            t = rand_timesteps(bt, "uniform", device=device)
            xt, z, target = sample_data_noise(x, t)
            v = model(xt, t, text, text_mask=text_mask)
            total += F.mse_loss(v.float(), target).item() * bt
            n += bt
    return total / max(n, 1), n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--promote", action="store_true",
                    help="copy ckpt to <ckpt-dir>/best.pt after a sane val")
    ap.add_argument("--max-mse", type=float, default=0.2,
                    help="refuse to promote if mse is above this")
    args = ap.parse_args()
    mse, n, el = validate(args.ckpt, args.data, args.device, batch=args.batch)
    print(json.dumps({"ckpt": args.ckpt, "flow_mse": mse, "n": n, "sec": round(el, 1)}))
    if args.promote:
        if mse > args.max_mse:
            raise SystemExit(f"mse {mse} above {args.max_mse}, NOT promoting")
        dst = Path(args.ckpt).parent / "best.pt"
        shutil.copy2(args.ckpt, dst)
        print(f"promoted -> {dst}")


if __name__ == "__main__":
    main()
