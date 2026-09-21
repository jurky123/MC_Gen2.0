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
    if len(batch[0]) == 3:
        xs = torch.stack([b[0] for b in batch], dim=0)
        if isinstance(batch[0][1], str):
            text = [b[1] for b in batch]
        else:
            text = torch.stack([b[1] for b in batch], dim=0)
        aux = {k: torch.stack([b[2][k] for b in batch], dim=0) for k in batch[0][2]}
        return xs, text, aux
    xs = torch.stack([b[0] for b in batch], dim=0)
    if isinstance(batch[0][1], str):
        return xs, [b[1] for b in batch]
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
    if kind == "pairs":
        from data.pair_dataset import MmapPairDataset

        return MmapPairDataset(
            ref_mmap=src["ref_mmap"],
            target_mmap=src["target_mmap"],
            metadata=src["metadata"],
            splits=src["splits"],
            split=split,
            ref_size=int(src.get("ref_size", ds.get("ref_size", 64))),
            target_size=image_size,
            rgba_mode=src.get("rgba_mode", ds.get("rgba_mode", "straight")),
            prompt_col=src.get("prompt_col", "prompt"),
            ref_premultiplied=bool(src.get("ref_premultiplied", False)),
            zero_reference=bool(src.get("zero_reference", False)),
            shuffle_reference=bool(src.get("shuffle_reference", False)),
        )
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
        prompt_views=src.get("prompt_views", "") or ds.get("prompt_views", ""),
        prompt_cols=src.get("prompt_cols", None) or ds.get("prompt_cols", None),
        toroidal_col=src.get("toroidal_col", "") or ds.get("toroidal_col", ""),
        toroidal_values=src.get("toroidal_values", None) or ds.get("toroidal_values", None),
        rgba_mode=src.get("rgba_mode", ds.get("rgba_mode", "straight")),
        return_aux=bool(src.get("return_aux", ds.get("return_aux", False))),
        tileable_col=src.get("tileable_col", ds.get("tileable_col", "tileable")),
        zero_reference=bool(src.get("zero_reference", ds.get("zero_reference", False))),
        ref_size=int(src.get("ref_size", ds.get("ref_size", 64))),
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

        if getattr(train_cfg, "freeze_backbone", False):
            trainable_prefixes = ("text_proj", "text_null", "cross_blocks",
                                  "head", "head_norm", "t_embedder",
                                  "ref_encoder", "ref_adapters", "ref_cross_blocks")
            frozen = trainable = 0
            for name, param in self.model.named_parameters():
                if name.startswith(trainable_prefixes):
                    param.requires_grad_(True); trainable += param.numel()
                else:
                    param.requires_grad_(False); frozen += param.numel()
            print(f"freeze_backbone: trainable {trainable/1e6:.2f}M, frozen {frozen/1e6:.2f}M")

        # Dynamic token-level text tower for cross-attention conditioning.
        self.text_encoder = None
        if getattr(model_cfg, "text_injection", "joint") == "cross_attn":
            from data.text_tower import get_text_encoder

            tower = train_cfg.text_tower or {}
            tower_device = tower.get("device") or self.device
            self.text_encoder = get_text_encoder(
                tower.get("model_name", ""),
                device=tower_device,
                dtype=tower.get("dtype", "bfloat16"),
                max_length=int(tower.get("max_length", model_cfg.max_text_tokens)),
                revision=tower.get("revision") or None,
                instruction=tower.get("instruction", ""),
                layers=tower.get("layers"),
                pad_bucket=tower.get("pad_bucket", 0),
            )
            print(f"text tower={tower.get('model_name')} dim={self.text_encoder.dim} "
                  f"layers={self.text_encoder.layers} len={self.text_encoder.max_length} "
                  f"device={tower_device}")

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

    def compute_loss(self, x, text, aux=None, text_pair=None):
        b = x.shape[0]
        t = rand_timesteps(b, self.tcfg.flow.get("timestep_sampling", "uniform"), device=x.device)
        xt, z, target = sample_data_noise(x, t)
        text_mask = None
        if text_pair is not None:
            text, text_mask = text_pair
        elif isinstance(text, (list, tuple)):
            text, text_mask = self.text_encoder.encode(list(text))
            text = text.to(self.device)
            text_mask = text_mask.to(self.device)
        if isinstance(text, torch.Tensor) and text.abs().sum().item() == 0.0:
            drop = 1.0
        else:
            drop = self.cond_drop
        tileable_mask = aux.get("tileable") if isinstance(aux, dict) else None
        reference = aux.get("reference") if isinstance(aux, dict) else None
        ref_drop = float(getattr(self.tcfg, "ref_dropout", 0.0)) if reference is not None else None
        if self.autocast is not None:
            with self.autocast:
                v = self.model(xt, t, text, text_mask=text_mask, cond_drop_prob=drop,
                               reference=reference, ref_drop_prob=ref_drop)
                return flow_tile_loss(v.float(), xt, t, target, self.tcfg.tile_loss,
                                      tileable_mask=tileable_mask)
        v = self.model(xt, t, text, text_mask=text_mask, cond_drop_prob=drop,
                       reference=reference, ref_drop_prob=ref_drop)
        return flow_tile_loss(v, xt, t, target, self.tcfg.tile_loss,
                              tileable_mask=tileable_mask)

    def _optimizer_step(self, tail_scale=1.0):
        # `tail_scale` > 1 compensates an epoch-tail step whose window holds
        # fewer than `accum` batches, so its gradient magnitude matches a full
        # accumulation window (P2-2) instead of being silently underweighted.
        if tail_scale != 1.0:
            for p in self._checkpoint_model().parameters():
                if p.grad is not None:
                    p.grad.mul_(tail_scale)
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

    def _conditioning_manifest(self):
        """Snapshot of everything that defines the conditioning pathway (P1-3)."""
        tower = self.tcfg.text_tower or {}
        manifest = {
            "text_injection": getattr(self.mcfg, "text_injection", "joint"),
            "text_dim": int(self.mcfg.text_dim),
            "max_text_tokens": int(self.mcfg.max_text_tokens),
            "cond_dropout": float(self.mcfg.cond_dropout),
            "channels": int(self.mcfg.in_channels),
            "image_size": int(self.mcfg.image_size),
            "rgba_mode": getattr(self, "rgba_mode", "") or "straight",
            "text_encoder": tower.get("model_name", ""),
            "text_encoder_revision": tower.get("revision") or "",
            "text_layers": list(tower.get("layers") or []),
            "text_instruction": tower.get("instruction", ""),
            "text_max_length": int(tower.get("max_length", self.mcfg.max_text_tokens)),
            "text_dtype": tower.get("dtype", "bfloat16"),
        }
        return manifest

    @staticmethod
    def _check_manifest(loaded, current, path):
        if not loaded:
            return
        diffs = {k: (loaded.get(k), current.get(k)) for k in current
                 if loaded.get(k) != current.get(k)}
        if diffs:
            raise ValueError(
                f"conditioning manifest mismatch vs {path}: {diffs}")

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
            "conditioning": self._conditioning_manifest(),
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
        # P1-4: the EMA shadow created in __init__ tracks the *fresh random*
        # model. After re-initialising the weights it must be rebuilt from the
        # current model, otherwise the shadow keeps anchoring at random init
        # (with high decay this permanently corrupts the average and makes the
        # EMA weights functionally broken).
        if self.ema is not None:
            ema_cfg = self.tcfg.ema
            self.ema = EMA(self._checkpoint_model(),
                           decay=float(ema_cfg.get("decay", 0.999)),
                           update_every=int(ema_cfg.get("update_every", 1)),
                           device=ema_cfg.get("device", "cpu"))
            print("EMA re-initialised from init-from weights")

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self._check_manifest(sd.get("conditioning"), self._conditioning_manifest(), path)
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
        self.data_cfg_path = data_cfg_path
        self.rgba_mode = str(ds_cfg.get("rgba_mode", "straight"))
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
        chunked = getattr(self.tcfg, "pipeline_encode", True) and self.text_encoder is not None
        if chunked:
            from .pipeline import ChunkedEncodedLoader

            tower_dev = (self.tcfg.text_tower or {}).get("device") or self.device
            pipeline = ChunkedEncodedLoader(
                train_loader, self.text_encoder, accum,
                tower_device=tower_dev, main_device=self.device)
            print("text-encode pipeline: ON "
                  f"(chunk={accum} micro-batches, tower={tower_dev})")
        while self.global_step < self.tcfg.steps:
            self.model.train()
            pending = 0
            pending_loss = 0.0
            source = pipeline if chunked else train_loader
            for item in source:
                if chunked:
                    items = item  # chunk: list of (x, text_pair, aux)
                else:
                    items = [item]  # single micro-batch
                for batch in items:
                    if chunked:
                        x, text_pair, aux = batch
                        text = None
                    else:
                        x, text = batch[0], batch[1]
                        aux = batch[2] if len(batch) > 2 else None
                        text_pair = None
                        if isinstance(text, torch.Tensor):
                            text = text.to(self.device, non_blocking=True)
                    x = x.to(self.device, non_blocking=True)
                    if isinstance(aux, dict):
                        aux = {k: v.to(self.device, non_blocking=True) for k, v in aux.items()}
                    loss = self.compute_loss(x, text, aux=aux, text_pair=text_pair)
                    scaled = loss / accum
                    if self.scaler is not None:
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    pending_loss += loss.detach().item()
                    pending += 1
                    if pending < accum:
                        continue
                    last_loss = pending_loss / pending
                    pending = 0
                    pending_loss = 0.0
                    self._optimizer_step()
                    if self.global_step % self.tcfg.log_every == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        el = time.time() - self.start_time
                        print(f"step {self.global_step}/{self.tcfg.steps} loss {last_loss:.5f} lr {lr:.2e} elapsed {el:.1f}s")
                    if self.tcfg.save_every and self.global_step % self.tcfg.save_every == 0:
                        self.save(out_dir / "latest.pt")
                    if self.global_step >= self.tcfg.steps:
                        break
                if self.global_step >= self.tcfg.steps:
                    break
            if chunked:
                pipeline.close()
            if pending > 0 and self.global_step < self.tcfg.steps:
                self._optimizer_step(tail_scale=accum / pending)
                if self.tcfg.save_every and self.global_step % self.tcfg.save_every == 0:
                    self.save(out_dir / "latest.pt")
            self.completed_epochs += 1
            self.save(out_dir / "latest.pt")
            if len(train_loader) > 0 and self.completed_epochs % self.tcfg.eval_every_epochs == 0:
                metrics = self.val_validate(val_ds, out_dir)
                score = metrics.get(self.tcfg.select_metric or "flow_mse", metrics.get("flow_mse"))
                if self.tcfg.save_best and score < best_val:
                    best_val = score
                    self.save(out_dir / "best.pt")
                    print(f"new best val {self.tcfg.select_metric or 'flow_mse'} {score:.5f} -> {out_dir / 'best.pt'}")
        self.save(out_dir / "latest.pt")

    def val_validate(self, val_ds, out_dir):
        self.model.eval()
        total = 0.0
        n = 0
        loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=_collate)
        with torch.no_grad():
            for batch in loader:
                x, text = batch[0], batch[1]
                aux = batch[2] if len(batch) > 2 else None
                x = x.to(self.device)
                if isinstance(aux, dict):
                    aux = {k: v.to(self.device) for k, v in aux.items()}
                reference = aux.get("reference") if isinstance(aux, dict) else None
                text_mask = None
                if isinstance(text, (list, tuple)):
                    text, text_mask = self.text_encoder.encode(list(text))
                    text = text.to(self.device)
                    text_mask = text_mask.to(self.device)
                else:
                    text = text.to(self.device)
                b = x.shape[0]
                t = rand_timesteps(b, "uniform", device=self.device)
                xt, z, target = sample_data_noise(x, t)
                if self.autocast is not None:
                    with self.autocast:
                        v = self.model(xt, t, text, text_mask=text_mask,
                                       reference=reference)
                else:
                    v = self.model(xt, t, text, text_mask=text_mask,
                                   reference=reference)
                total += F.mse_loss(v.float(), target).item() * b
                n += b
        mse = total / max(n, 1)
        print(f"[val] step {self.global_step} mse {mse:.5f}")
        return {"flow_mse": mse}


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
