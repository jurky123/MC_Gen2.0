"""Deterministic step-addressed training and validation; atomic resumable checkpoints."""
from contextlib import nullcontext
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F

from .catalog import read_catalog, atomic_json, file_hash
from .conditioning import TextEncoder
from .config import Experiment
from .dataset import CatalogDataset
from .model import PixelFlow
from .tasks import TaskSampler, conditioned_prompts
from .runtime import load_checkpoint, save_checkpoint


def flow_loss(predicted, image, noisy, t, noise, tileable, seam_weight):
    mse = F.mse_loss(predicted.float(), (noise - image).float())
    seam = mse.new_zeros(())
    active = tileable & (t < 0.5)
    if seam_weight and active.any():
        reconstructed = noisy - t[:, None, None, None] * predicted
        x = reconstructed[active]
        seam = ((x[..., 0] - x[..., -1]).abs().mean() +
                (x[..., 0, :] - x[..., -1, :]).abs().mean()) / 2
    return mse + seam_weight * seam


class Trainer:
    def __init__(self, config, catalog, output, device="cpu", text_device=None, resume=None, initialize=None, encoder=None):
        if config.conditioning_input == 'visual_tokens':
            script = 'train_curated_mixed.py' if config.train.trainable == 'joint_finetune' else 'train_visual_reference.py'
            raise ValueError(f'use scripts/{script} for this spatial-reference training protocol')
        if config.conditioning_input == "multimodal_embedding":
            raise ValueError("use scripts/train_reference_adapter.py for pooled multimodal adapter training")
        if resume and initialize:
            raise ValueError("resume and initialize are mutually exclusive")
        self.cfg, self.device = config, device
        self.out = Path(output)
        if self.out.exists() and any(self.out.iterdir()) and not resume:
            raise ValueError("output directory is not empty; use a new run or explicit resume")
        records, self.meta = read_catalog(catalog)
        if self.meta["native_size"] != config.model.size:
            raise ValueError("catalog/model native size mismatch")
        kwargs = dict(size=config.model.size, ref_size=config.model.ref_size,
                      asset_type=config.asset_type, task=config.task, alpha_mode=config.alpha_mode, style=config.style, target_scope=config.target_scope)
        self.train_data = CatalogDataset(records, split="train", **kwargs)
        self.val_data = CatalogDataset(records, split="val", **kwargs)
        if config.task in ("reference", "joint") and not any(r.get("reference") for r in self.train_data.records):
            raise ValueError("reference/joint training requires reference examples")
        self.sampler = TaskSampler(self.train_data, config)
        torch.manual_seed(config.train.seed)
        self.model = PixelFlow(config.model).to(device)
        # Byte embeddings are exclusively for the explicit lightweight baseline.
        if config.text.backend == "hf" and config.model.byte_tokens:
            self.model.byte_embedding.requires_grad_(False)
            self.model.byte_position.requires_grad_(False)
        self.encoder = encoder if encoder is not None else TextEncoder(config.text, config.model, text_device or device)
        self.initialization = None
        if initialize:
            state = load_checkpoint(initialize)
            source = Experiment.from_dict(state["config"])
            before, after = source.to_dict()["model"], config.to_dict()["model"]
            # Deliberate architecture extension; arbitrary partial loads are forbidden.
            before.pop("reference_encoding"); after.pop("reference_encoding")
            if before != after or state["text_identity"] != self.encoder.identity:
                raise ValueError("initialize requires compatible model and identical text encoder")
            if source.asset_type != config.asset_type or source.alpha_mode != config.alpha_mode or source.style != config.style:
                raise ValueError("initialize requires matching output contract")
            extension = source.model.reference_encoding == "spatial" and config.model.reference_encoding == "dual"
            if source.model.reference_encoding != config.model.reference_encoding and not extension:
                raise ValueError("unsupported reference architecture initialization")
            result = self.model.load_state_dict(state["model"], strict=not extension)
            if extension:
                expected = {n for n in self.model.state_dict() if self.is_attribute(n)}
                if set(result.missing_keys) != expected or result.unexpected_keys:
                    raise ValueError("unexpected keys in attribute adapter initialization")
            self.initialization = {"path": str(Path(initialize).resolve()), "sha256": file_hash(initialize),
                                   "step": state["step"], "catalog_fingerprint": state["catalog_fingerprint"]}
        if config.train.trainable == "attributes":
            if not initialize and not resume:
                raise ValueError("attribute-only training requires an initialized backbone")
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(self.is_attribute(name))
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=config.train.lr, weight_decay=config.train.weight_decay)
        self.step, self.best = 0, float("inf")
        self.best_by_task = {}
        if resume:
            state = load_checkpoint(resume)
            if Experiment.from_dict(state["config"]).to_dict() != config.to_dict() or state["catalog_fingerprint"] != self.meta["fingerprint"]:
                raise ValueError("resume requires exactly matching config and catalog")
            if state["text_identity"] != self.encoder.identity:
                raise ValueError("text encoder artifacts changed since training")
            self.model.load_state_dict(state["model"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.step, self.best = state["step"], state["best"]
            self.best_by_task = state.get("best_by_task", {})
            self.initialization = state.get("initialization")
        self.out.mkdir(parents=True, exist_ok=True)
        atomic_json(self.out / "sampling_budget.json", self.sampler.report(config))
        atomic_json(self.out / "run.json", {"config": config.to_dict(), "catalog_fingerprint": self.meta["fingerprint"],
                                           "catalog": str(Path(catalog).resolve()), "device": str(device),
                                           "text_identity": self.encoder.identity,
                                           "initialization": self.initialization,
                                           "torch_version": str(torch.__version__)})

    @staticmethod
    def is_attribute(name):
        return name.startswith("attribute_encoder.") or ".attribute_attention." in name or name.endswith(".attribute_gate")

    def autocast(self):
        if self.cfg.train.precision == "bfloat16":
            return torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16)
        return nullcontext()

    def condition(self, batch, generator=None, mode=None):
        prompts = batch["prompts"]
        if mode == "reference" or self.cfg.task == "reference":
            prompts = [""] * len(prompts)
        elif "task_modes" in batch:
            prompts = [p if m in ("text", "joint") else "" for p, m in zip(prompts, batch["task_modes"])]
        text, mask = self.encoder.encode(conditioned_prompts(prompts, self.cfg), self.device)
        present = batch["ref_present"].clone()
        mask &= torch.tensor([bool(p.strip()) for p in batch["prompts"]], device=self.device)[:, None]
        if mode == "text" or self.cfg.task == "text":
            present[:] = False
        if mode == "reference" or self.cfg.task == "reference":
            mask[:] = False
        if "task_modes" in batch:
            mask &= torch.tensor([m in ("text", "joint") for m in batch["task_modes"]], device=self.device)[:, None]
            present &= torch.tensor([m in ("reference", "joint") for m in batch["task_modes"]], device=self.device)
        elif generator is not None:
            mask &= (torch.rand(mask.shape[0], generator=generator, device=self.device) >= self.cfg.train.text_dropout)[:, None]
            present &= torch.rand(present.shape[0], generator=generator, device=self.device) >= self.cfg.train.ref_dropout
        reference_mode = None
        if self.cfg.model.reference_encoding == "dual":
            reference_mode = ["attributes" if bool(m.any()) else "shape" for m in mask]
        return self.model.context(text, mask, batch["reference"], present, reference_mode=reference_mode)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        scores = {}
        # The same fixed examples/noise/timesteps every time, independent of
        # training RNG; separate text and reference scores cannot hide regressions.
        modes = (self.cfg.task,) if self.cfg.task != "joint" else ("text", "reference", "joint")
        for mode in modes:
            from .evaluation import select_indices
            available = [i for i, r in enumerate(self.val_data.records) if mode == "text" or r.get("reference")]
            if not available:
                raise ValueError(f"no validation examples for {mode}; inspect the global split")
            chosen = select_indices([self.val_data.records[i] for i in available], self.cfg.train.val_samples, 7919)
            order = [available[i] for i in chosen]
            n = len(order)
            gen = torch.Generator(device=self.device).manual_seed(104729)
            total = 0.
            for start in range(0, n, self.cfg.train.batch_size):
                batch = self.val_data.batch(order[start:start + self.cfg.train.batch_size], self.device)
                x = batch["image"]
                t = torch.rand(len(x), generator=gen, device=self.device)
                noise = torch.randn(x.shape, generator=gen, device=self.device)
                noisy = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
                with self.autocast():
                    pred = self.model(noisy, t, self.condition(batch, mode=mode))
                total += F.mse_loss(pred.float(), noise - x).item() * len(x)
            scores[mode + "_flow_mse"] = total / n
            scores[mode + "_samples"] = n
        self.model.train()
        return scores

    def save(self, name):
        save_checkpoint(self.out / name, {"config": self.cfg.to_dict(), "model": self.model.state_dict(),
                                         "optimizer": self.optimizer.state_dict(), "step": self.step,
                                         "best": self.best, "best_by_task": self.best_by_task, "catalog_fingerprint": self.meta["fingerprint"],
                                         "text_identity": self.encoder.identity, "initialization": self.initialization})

    def run(self, stop_after=None):
        cfg = self.cfg.train
        end = min(cfg.steps, stop_after) if stop_after is not None else cfg.steps
        self.model.train()
        for step in range(self.step, end):
            if getattr(self, 'stop_requested', False):
                self.save('latest.pt')
                atomic_json(self.out / 'stopped.json', {'step': self.step, 'reason': 'requested_stop', 'checkpoint': 'latest.pt'})
                break
            rng = torch.Generator().manual_seed(cfg.seed + 1000003 * step)
            gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 1000003 * step + 17)
            local_step = max(0, step - cfg.schedule_start)
            warm = min(1., (local_step + 1) / max(cfg.warmup, 1))
            progress = max(0., (local_step - cfg.warmup) / max(cfg.steps - cfg.schedule_start - cfg.warmup, 1))
            lr = cfg.lr * warm * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(cfg.accumulation):
                if cfg.task_mix:
                    indices, modes = self.sampler.sample(cfg.batch_size, rng)
                else:
                    indices = self.train_data.sample_indices(cfg.batch_size, rng, cfg.paired_fraction)
                    modes = None
                batch = self.train_data.batch(indices, self.device)
                if modes is not None:
                    batch["task_modes"] = modes
                x = batch["image"]
                t = torch.rand(len(x), generator=gen, device=self.device)
                noise = torch.randn(x.shape, generator=gen, device=self.device)
                noisy = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
                with self.autocast():
                    pred = self.model(noisy, t, self.condition(batch, generator=gen))
                    loss = flow_loss(pred, x, noisy, t, noise, batch["tileable"], cfg.seam_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at step {step}")
                (loss / cfg.accumulation).backward()
                total += loss.item() / cfg.accumulation
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.step = step + 1
            row = {"step": self.step, "loss": total, "lr": lr, "text_cache": self.encoder.cache_stats()}
            if getattr(self, 'stop_requested', False):
                with (self.out / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                self.save('latest.pt')
                atomic_json(self.out / 'stopped.json', {'step': self.step, 'reason': 'requested_stop', 'checkpoint': 'latest.pt'})
                break
            if self.step % cfg.eval_every == 0 or self.step == end:
                row.update(self.validate())
                # Flow-only selection is explicitly provisional. Generation
                # benchmark reports (both tasks) are required for promotion.
                score = sum(v for k, v in row.items() if k.endswith("_flow_mse")) / sum(k.endswith("_flow_mse") for k in row)
                improved = []
                best_mean = score < self.best
                if best_mean:
                    self.best = score
                for mode in ("text", "reference", "joint"):
                    value = row.get(mode + "_flow_mse")
                    if value is not None and value < self.best_by_task.get(mode, float("inf")):
                        self.best_by_task[mode] = value
                        improved.append(mode)
                if best_mean:
                    self.save("best_flow.pt")
                for mode in improved:
                    self.save(f"best_{mode}_flow.pt")
                self.save("latest.pt")
                print(json.dumps(row), flush=True)
            with (self.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        return self.out / "latest.pt"
