import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from config import ModelConfig, TrainConfig, load_yaml
from model.mc_flow_dit import MCFlowDiT
from data.build_mmap import MmapImageTextDataset, ManifestImageDataset, MixDataset
from .flow import rand_timesteps, sample_data_noise
from .losses import flow_tile_loss
from .ema import EMA


def _collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    embs = torch.stack([b[1] for b in batch], dim=0)
    return xs, embs


def _worker_init(worker_id):
    # Seed numpy per worker so toroidal rolls / text-view sampling differ
    # between DataLoader workers (forked workers otherwise share RNG state).
    import numpy as _np

    _np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def _build_source(src, ds, split):
    kind = src.get("type", "mmap")
    image_size = int(ds.get("image_size", 32))
    normalize = bool(ds.get("normalize", True))
    text_dim = int(ds.get("text_dim", 768))
    channels = int(src.get("channels", ds.get("channels", 4)))
    if kind == "manifest":
        return ManifestImageDataset(
            manifest=src["manifest"],
            image_size=image_size,
            channels=channels,
            normalize=normalize,
            split=split,
            text_dim=text_dim,
            toroidal=split == "train" and bool(src.get("toroidal", ds.get("toroidal", False))),
        )
    return MmapImageTextDataset(
        images=src["images"],
        metadata=src["metadata"],
        splits=src["splits"],
        split=split,
        image_size=image_size,
        toroidal=split == "train" and bool(src.get("toroidal", ds.get("toroidal", True))),
        normalize=normalize,
        text_mmap=src.get("text_mmap", ""),
        text_dim=text_dim,
        max_text_tokens=int(ds.get("max_text_tokens", 64)),
        channels=channels,
        text_views=int(src.get("text_views", ds.get("text_views", 0))),
    )


def build_dataset(data_cfg_path, split, train_cfg):
    data = load_yaml(data_cfg_path)
    ds = data.get("dataset", {})
    sources = ds.get("sources")
    if sources:
        datasets = [_build_source(src, ds, split) for src in sources]
        weights = [float(src.get("weight", 1.0)) for src in sources]
        return datasets[0] if len(datasets) == 1 else MixDataset(datasets, weights)
    return _build_source(ds, ds, split)


def get_lr_schedule(optimizer, steps, warmup):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def probe_batch_size(model, data, text_cfg, train_cfg, device):
    cfg = train_cfg
    batch_cfg = cfg.batch
    if not batch_cfg.get("auto_probe", True) or not torch.cuda.is_available():
        return int(batch_cfg.get("preferred_micro_batch", 16))
    bs = int(batch_cfg.get("preferred_micro_batch", 128))
    max_gb = float(batch_cfg.get("max_vram_gb", 7.2))
    base = getattr(model, "_orig_mod", model)
    in_channels = base.cfg.in_channels
    size = base.cfg.image_size
    text_dim = text_cfg.get("text_dim", 768)
    max_tokens = text_cfg.get("max_text_tokens", 64)
    while bs >= 1:
        try:
            x = torch.randn(bs, in_channels, size, size, device=device)
            t = rand_timesteps(bs, device=device)
            text = torch.randn(bs, max_tokens, text_dim, device=device)
            v = model(x, t, text)
            (v * v).mean().backward()
            model.zero_grad(set_to_none=True)
            used = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            if used <= max_gb:
                return bs
            bs = bs // 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = bs // 2
    return 1


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, device=None):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(train_cfg.seed)

        self.model = MCFlowDiT(model_cfg)
        self.cond_drop = model_cfg.cond_dropout
        self.model.to(self.device)

        self.compile_requested = bool(train_cfg.compile)
        if self.compile_requested and not hasattr(torch, "compile"):
            print("torch.compile unavailable, skipping")
            self.compile_requested = False

        opt_cfg = train_cfg.optimizer
        fused = bool(opt_cfg.get("fused", True)) and torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg.get("lr", 3e-4)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            weight_decay=float(opt_cfg.get("weight_decay", 0.03)),
            fused=fused,
        )
        self.scheduler = get_lr_schedule(self.optimizer, train_cfg.steps, int(train_cfg.scheduler.get("warmup_steps", 1000)))

        ema_cfg = train_cfg.ema
        self.ema = None
        if ema_cfg.get("enabled", True):
            self.ema = EMA(self.model, decay=float(ema_cfg.get("decay", 0.999)), update_every=int(ema_cfg.get("update_every", 1)), device=ema_cfg.get("device", "cpu"))

        self.autocast = None
        self.scaler = None
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float16
            self.autocast = torch.autocast(device_type="cuda", dtype=dtype)
            if dtype == torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()

        self.global_step = 0
        self.completed_epochs = 0
        self.start_time = time.time()

    def _dtype_text(self):
        return torch.float32

    def compute_loss(self, x, text):
        b = x.shape[0]
        t = rand_timesteps(b, self.tcfg.flow.get("timestep_sampling", "uniform"), device=x.device)
        xt, z, target = sample_data_noise(x, t)
        drop = 1.0 if text.abs().sum().item() == 0.0 else self.cond_drop
        if self.autocast is not None:
            with self.autocast:
                v = self.model(xt, t, text, cond_drop_prob=drop)
                return flow_tile_loss(v.float(), xt, t, target, self.tcfg.tile_loss)
        v = self.model(xt, t, text, cond_drop_prob=drop)
        return flow_tile_loss(v, xt, t, target, self.tcfg.tile_loss)

    def _optimizer_step(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            self._grad_clip()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self._grad_clip()
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        if self.ema is not None:
            self.ema.step(self._checkpoint_model())

    def _checkpoint_model(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _grad_clip(self):
        if self.tcfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg.grad_clip))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = {
            "model": self._checkpoint_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "model_cfg": self.mcfg.to_dict(),
            "train_cfg": self.tcfg.to_dict(),
        }
        if self.ema is not None:
            sd["ema"] = self.ema.state_dict()
        sd["completed_epochs"] = self.completed_epochs
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(sd, str(tmp_path))
        os.replace(tmp_path, path)
        print(f"checkpoint saved: {path}")

    def init_from(self, path, use_ema=False):
        """Initialise model weights from another checkpoint without resuming.

        Tensors whose name+shape match are copied; new or shape-mismatched ones
        (e.g. ``text_proj`` when switching to a new text dimension) stay at their
        fresh init. Optimizer / scheduler / step are *not* restored.
        """
        sd = torch.load(path, map_location="cpu")
        src = sd["ema"]["shadow"] if use_ema and "ema" in sd else sd["model"]
        dst = self._checkpoint_model().state_dict()
        copied, skipped = 0, []
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k].copy_(v.to(dtype=dst[k].dtype))
                copied += 1
            else:
                skipped.append(k)
        print(f"init-from {path} ({'ema' if use_ema else 'model'}): copied {copied} tensors"
              f", skipped {len(skipped)}")
        if skipped:
            print("  skipped:", skipped)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self._checkpoint_model().load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.global_step = sd["step"]
        self.completed_epochs = int(sd.get("completed_epochs", 0))
        if self.ema is not None and "ema" in sd:
            self.ema.load_state_dict(sd["ema"])
        if self.tcfg.scheduler.get("reset_on_resume", False):
            remaining = max(1, int(self.tcfg.steps) - self.global_step)
            warmup = min(int(self.tcfg.scheduler.get("warmup_steps", 0)), remaining)
            self.scheduler = get_lr_schedule(self.optimizer, remaining, warmup)
            print(f"scheduler reset on resume: {remaining} remaining steps, warmup {warmup}")
        print(f"resumed from {path} at step {self.global_step}, completed_epochs {self.completed_epochs}")

    def train(self, data_cfg_path):
        ds_cfg = load_yaml(data_cfg_path).get("dataset", {})
        text_cfg = {"text_dim": ds_cfg.get("text_dim", 768), "max_text_tokens": ds_cfg.get("max_text_tokens", 64)}
        bs = probe_batch_size(self.model, None, text_cfg, self.tcfg, self.device)
        if self.compile_requested and torch.cuda.is_available():
            print("compiling training graph after batch-size probe")
            self.model = torch.compile(self.model)
        accum = max(1, int(self.tcfg.gradient_accumulation))
        print(f"device={self.device} micro_batch={bs} grad_accum={accum} effective_batch={bs * accum}")

        train_ds = build_dataset(data_cfg_path, "train", self.tcfg)
        val_ds = build_dataset(data_cfg_path, "val", self.tcfg)
        sampler = train_ds.make_sampler() if hasattr(train_ds, "make_sampler") else None
        num_workers = int(self.tcfg.batch.get("num_workers", 8))
        loader_kwargs = dict(
            dataset=train_ds,
            batch_size=bs,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(self.tcfg.batch.get("prefetch_factor", 4))
            loader_kwargs["worker_init_fn"] = _worker_init
        train_loader = torch.utils.data.DataLoader(**loader_kwargs)
        print(f"train samples={len(train_ds)} val samples={len(val_ds)} "
              f"dataloader_workers={num_workers}")

        out_dir = Path(self.tcfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer.zero_grad(set_to_none=True)
        best_val = float("inf")
        while self.global_step < self.tcfg.steps:
            self.model.train()
            pending = 0
            last_loss = 0.0
            for x, text in train_loader:
                x = x.to(self.device, non_blocking=True)
                text = text.to(self.device, non_blocking=True)
                loss = self.compute_loss(x, text)
                scaled = loss / accum
                if self.scaler is not None:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                last_loss = loss.detach().item()
                pending += 1
                if pending < accum:
                    continue
                pending = 0
                self._optimizer_step()
                if self.global_step % self.tcfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    el = time.time() - self.start_time
                    print(f"step {self.global_step}/{self.tcfg.steps} loss {last_loss:.5f} lr {lr:.2e} elapsed {el:.1f}s")
                if self.tcfg.save_every and self.global_step % self.tcfg.save_every == 0:
                    self.save(out_dir / "latest.pt")
                if self.global_step >= self.tcfg.steps:
                    break
            if pending > 0 and self.global_step < self.tcfg.steps:
                self._optimizer_step()
                if self.tcfg.save_every and self.global_step % self.tcfg.save_every == 0:
                    self.save(out_dir / "latest.pt")
            self.completed_epochs += 1
            self.save(out_dir / "latest.pt")
            if len(train_loader) > 0 and self.completed_epochs % self.tcfg.eval_every_epochs == 0:
                mse = self.val_validate(val_ds, out_dir)
                if self.tcfg.save_best and mse < best_val:
                    best_val = mse
                    self.save(out_dir / "best.pt")
                    print(f"new best val mse {mse:.5f} -> {out_dir / 'best.pt'}")
        self.save(out_dir / "latest.pt")

    def val_validate(self, val_ds, out_dir):
        self.model.eval()
        total = 0.0
        n = 0
        loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=_collate)
        with torch.no_grad():
            for x, text in loader:
                x = x.to(self.device)
                text = text.to(self.device)
                b = x.shape[0]
                t = rand_timesteps(b, "uniform", device=self.device)
                xt, z, target = sample_data_noise(x, t)
                v = self.model(xt, t, text)
                total += F.mse_loss(v, target).item() * b
                n += b
        mse = total / max(n, 1)
        print(f"[val] step {self.global_step} mse {mse:.5f}")
        return mse


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model/base.yaml")
    ap.add_argument("--train", default="configs/train/smoke.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-from", default="", help="initialise weights from a checkpoint (no optimizer state)")
    ap.add_argument("--init-from-ema", action="store_true", help="use EMA shadow weights for --init-from")
    ap.add_argument("--device", default="")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    mcfg = ModelConfig.from_yaml(args.model)
    tcfg = TrainConfig.from_yaml(args.train)

    log_path = args.log_file or tcfg.log_file or str(Path(tcfg.output_dir) / "train.log")
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "a", buffering=1, encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_handle)
    sys.stderr = _Tee(orig_err, log_handle)
    print(f"logging to {log_file}")

    trainer = Trainer(mcfg, tcfg, device=args.device or None)
    print(f"model={mcfg.name} params={trainer.model.param_count() / 1e6:.2f}M")
    if args.resume:
        trainer.load(args.resume)
    elif args.init_from:
        trainer.init_from(args.init_from, use_ema=args.init_from_ema)
    trainer.train(tcfg.dataset)


if __name__ == "__main__":
    main()
