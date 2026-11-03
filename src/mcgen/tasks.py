"""Explicit task distributions and target-level sampling, independent of CFG dropout."""
from collections import defaultdict
import torch

MODES = ("text", "reference", "joint", "unconditional")


def conditioned_prompts(prompts, config):
    # A specialist checkpoint has one output contract. Empty captions remain
    # empty: neither metadata nor an instruction may turn CFG null into text.
    prefix = "; ".join(x for x in ("Minecraft pixel art", config.asset_type if config.asset_type != "all" else "",
                                  config.style) if x)
    return [f"{prefix}: {p}" if p.strip() else "" for p in prompts]


class TaskSampler:
    def __init__(self, dataset, config):
        self.mix = config.train.task_mix
        self.dataset = dataset
        all_targets, paired = defaultdict(list), defaultdict(list)
        for i, r in enumerate(dataset.records):
            all_targets[r["target_hash"]].append(i)
            if r.get("reference"):
                paired[r["target_hash"]].append(i)
        self.pools = {"text": list(all_targets.values()), "reference": list(paired.values()),
                      "joint": list(paired.values()),
                      "unconditional": list((paired if config.task == "reference" else all_targets).values())}
        if self.mix:
            for mode in MODES:
                if self.mix[mode] and not self.pools[mode]:
                    raise ValueError(f"no training targets for requested task {mode}")
            self.weights = torch.tensor([self.mix[m] for m in MODES], dtype=torch.float64)

    def sample(self, n, generator):
        modes = torch.multinomial(self.weights, n, replacement=True, generator=generator).tolist()
        indices, names = [], []
        for m in modes:
            mode = MODES[m]
            pool = self.pools[mode]
            group = pool[int(torch.randint(len(pool), (), generator=generator))]
            indices.append(group[int(torch.randint(len(group), (), generator=generator))])
            names.append(mode)
        return indices, names

    def report(self, config):
        draws = config.train.steps * config.train.batch_size * config.train.accumulation
        if not self.mix:
            return {"policy": "legacy_dropout", "warning": "implicit task distribution; use task_mix for new runs"}
        return {"policy": "explicit_task_mix", "total_draws": draws,
                "tasks": {m: {"probability": self.mix[m], "unique_targets": len(self.pools[m]),
                              "expected_draws": draws * self.mix[m],
                              "expected_draws_per_target": draws * self.mix[m] / len(self.pools[m]) if self.pools[m] else 0}
                          for m in MODES},
                "note": "Per-task exposure; targets shared by tasks receive the sum. No silent pool fallback."}
