"""Public command line. All paths are explicit; legacy scripts remain separate."""
import argparse
import json
from pathlib import Path

from .catalog import read_catalog, atomic_json
from .config import Experiment


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mcgen")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("benchmark", help="authored semantic cases with blind human review")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-device")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p = commands.add_parser("review", help="summarize completed human benchmark ratings")
    p.add_argument("--directory", required=True)
    p = commands.add_parser("prepare", help="project holdouts, duplicate quarantine and native target compaction")
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project-plan", help="JSON project-to-split assignments (unspecified projects use stable hashing)")
    p.add_argument("--no-compact", action="store_true")
    p = commands.add_parser("budget", help="inspect task exposure before loading Qwen or starting training")
    p.add_argument("--config", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out")
    p = commands.add_parser("import-images", help="build a catalog from an explicit PNG JSONL manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, choices=(16, 32, 64), default=16)
    p.add_argument("--ref-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("merge", help="merge sources and regenerate one global split")
    p.add_argument("--catalog", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("migrate", help="issue a fresh catalog from retained v4 pairs")
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, choices=(16, 32), default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-text", action="store_true", help="include the full text dataset in the same global split")
    p = commands.add_parser("audit", help="verify full catalog lineage and source hashes")
    p.add_argument("--catalog", required=True)
    p = commands.add_parser("audit-legacy", help="report legacy contamination without changing artifacts")
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--out", required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume")
    p.add_argument("--initialize", help="weights-only transfer; new optimizer and step count")
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-device")
    p = commands.add_parser("sample")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--reference")
    p.add_argument("--conditioning-mode", choices=("native", "multimodal_embedding", "visual_tokens"), default="native",
                   help="reference input protocol; visual_tokens checkpoints support independent text/ref scales")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--text-scale", type=float, default=2.)
    p.add_argument("--ref-scale", type=float, default=2.)
    p.add_argument("--reference-mode", choices=("attributes", "shape"), help="dual checkpoints: default attributes with text, shape for image-only")
    p.add_argument("--solver", choices=("euler", "heun"), default="heun")
    p.add_argument("--export-scale", type=int, default=1)
    p.add_argument("--palette-colors", type=int, default=0)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-device")
    p.add_argument("--target-scope", choices=("all", "paired"), help="default: checkpoint scope; use all to test broader generalization")
    p.add_argument("--training-catalog", help="required for external benchmark overlap audit")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p = commands.add_parser("serve")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text-device")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = commands.add_parser("smoke", help="CPU-only synthetic end-to-end verification")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(args.checkpoint, args.manifest, args.out, device=args.device,
                      text_device=args.text_device, steps=args.steps, seeds=args.seeds)
    elif args.command == "review":
        from .benchmark import summarize_review
        print(json.dumps(summarize_review(args.directory), indent=2))
    elif args.command == "prepare":
        from .prepare import prepare_catalog
        plan = json.loads(Path(args.project_plan).read_text()) if args.project_plan else None
        print(json.dumps(prepare_catalog(args.catalog, args.out, args.seed, plan, not args.no_compact), indent=2))
    elif args.command == "budget":
        from .dataset import CatalogDataset
        from .tasks import TaskSampler
        cfg = Experiment.load(args.config)
        records, meta = read_catalog(args.catalog)
        ds = CatalogDataset(records, cfg.model.size, cfg.model.ref_size, "train", cfg.asset_type, cfg.task, cfg.alpha_mode, cfg.style, cfg.target_scope)
        report = {"catalog_fingerprint": meta["fingerprint"], "config": cfg.to_dict(),
                  "sampling": TaskSampler(ds, cfg).report(cfg)}
        if args.out:
            if Path(args.out).exists():
                parser.error("budget output already exists")
            atomic_json(args.out, report)
        print(json.dumps(report, indent=2))
    elif args.command == "import-images":
        from .ingest import import_images
        print(json.dumps(import_images(args.manifest, args.out, args.size, args.ref_size, args.seed), indent=2))
    elif args.command == "merge":
        from .ingest import merge_catalogs
        print(json.dumps(merge_catalogs(args.catalog, args.out, args.seed), indent=2))
    elif args.command == "migrate":
        from .migrate import migrate_pairs
        print(json.dumps(migrate_pairs(args.project, args.out, args.size, args.seed, include_text=args.include_text), indent=2))
    elif args.command == "audit":
        _, meta = read_catalog(args.catalog)
        print(json.dumps(meta, indent=2))
    elif args.command == "audit-legacy":
        from .migrate import legacy_audit
        print(json.dumps(legacy_audit(args.project, args.out), indent=2))
    elif args.command == "train":
        from .training import Trainer
        cfg = Experiment.load(args.config)
        if cfg.text.backend == "byte":
            print("BYTE BACKEND: pipeline/architecture baseline, no pretrained language semantics.", flush=True)
        Trainer(cfg, args.catalog, args.out, args.device, args.text_device, args.resume, args.initialize).run()
    elif args.command == "sample":
        from PIL import Image
        from .runtime import Runtime
        from .images import export, apply_alpha
        if not args.prompt.strip() and not args.reference:
            parser.error("provide --prompt or --reference")
        out = Path(args.out)
        if out.exists() or out.with_suffix(out.suffix + ".json").exists():
            parser.error("sample output already exists")
        runtime = Runtime(args.checkpoint, args.device, args.text_device)
        ref = Image.open(args.reference).convert("RGBA") if args.reference else None
        image = runtime.generate([args.prompt], [ref] if ref else None, seeds=[args.seed], steps=args.steps,
                                 conditioning_mode=args.conditioning_mode,
                                 text_scale=args.text_scale, ref_scale=args.ref_scale, solver=args.solver,
                                 reference_mode=args.reference_mode)[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        export(apply_alpha(image, runtime.config.alpha_mode), args.export_scale, args.palette_colors).save(out)
        atomic_json(out.with_suffix(out.suffix + ".json"), {**runtime.metadata, "request": vars(args)})
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.checkpoint, args.catalog, args.out, n=args.n, seed=args.seed, split=args.split,
                 device=args.device, text_device=args.text_device, steps=args.steps, training_catalog=args.training_catalog, target_scope=args.target_scope)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.checkpoint, args.device, args.text_device), host=args.host, port=args.port)
    elif args.command == "smoke":
        import torch
        from .smoke import fixture, tiny_config
        from .training import Trainer
        from .evaluation import evaluate
        torch.set_num_threads(1)
        out = Path(args.out)
        if out.exists():
            parser.error("smoke output already exists")
        catalog = fixture(out / "fixture")
        checkpoint = Trainer(tiny_config(), catalog, out / "train").run()
        evaluate(checkpoint, catalog, out / "evaluation", n=3, steps=2)
        print(f"smoke completed: {out} (synthetic fixture; no quality claim)")


if __name__ == "__main__":
    main()
