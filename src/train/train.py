import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from config import ModelConfig, TrainConfig, load_yaml
from model.mc_flow_dit import MCFlowDiT
from data.build_mmap import MmapImageTextDataset
from .flow import rand_timesteps, sample_data_noise
from .losses import flow_tile_loss
from .ema import EMA


def _collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    embs = torch.stack([b[1] for b in batch], dim=0)
    return xs, embs


def build_dataset(data_cfg_path, split, train_cfg):
    data = load_yaml(data_cfg_path)
    ds = data.get("dataset", {})
    return MmapImageTextDataset(
        images=ds["images"],
        metadata=ds["metadata"],
        splits=ds["splits"],
        split=split,
        image_size=int(ds.get("image_size", 32)),
        toroidal=bool(ds.get("toroidal", True)),
        normalize=bool(ds.get("normalize", True)),
        text_mmap=ds.get("text_mmap", ""),
        text_dim=int(ds.get("text_dim", 768)),
        max_text_tokens=int(ds.get("max_text_tokens", 64)),
        channels=int(ds.get("channels", 3)),
    )


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
            self.ema = EMA(self.model, decay=float(ema_cfg.get("decay", 0.9999)), update_every=int(ema_cfg.get("update_every", 8)), device=ema_cfg.get("device", "cpu"))

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

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self._checkpoint_model().load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.global_step = sd["step"]
        self.completed_epochs = int(sd.get("completed_epochs", 0))
        if self.ema is not None and "ema" in sd:
            self.ema.load_state_dict(sd["ema"])
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
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=_collate,
        )
        print(f"train samples={len(train_ds)} val samples={len(val_ds)}")

        out_dir = Path(self.tcfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer.zero_grad(set_to_none=True)
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
                self.val_validate(val_ds, out_dir)
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
        print(f"[val] step {self.global_step} mse {total / max(n, 1):.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model/base.yaml")
    ap.add_argument("--train", default="configs/train/smoke.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    mcfg = ModelConfig.from_yaml(args.model)
    tcfg = TrainConfig.from_yaml(args.train)
    trainer = Trainer(mcfg, tcfg, device=args.device or None)
    print(f"model={mcfg.name} params={trainer.model.param_count() / 1e6:.2f}M")
    if args.resume:
        trainer.load(args.resume)
    trainer.train(tcfg.dataset)


if __name__ == "__main__":
    main()
