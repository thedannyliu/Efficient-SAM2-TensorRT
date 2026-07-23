from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import summarize_trace
from .build import build_bundle
from .export import export_bundle
from .manifest import BundleManifest
from .lock import pin_environment
from .model_registry import load_registry, resolve_model
from .probe import write_probe
from .validate import accuracy_gate, write_gate_result
from .trt_benchmark import benchmark_engine, write_engine_benchmark
from .pytorch_benchmark import benchmark_pytorch_graphs, write_pytorch_graph_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam2-trt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("list-models", help="list configured model IDs")
    models.add_argument("--registry")

    probe = subparsers.add_parser("probe", help="record the deployment environment")
    probe.add_argument("--output", required=True)

    pin = subparsers.add_parser("pin", help="pin versions from a Thor probe")
    pin.add_argument("--probe", required=True)
    pin.add_argument("--output", required=True)

    export = subparsers.add_parser("export", help="export strongly typed ONNX graphs")
    export.add_argument("--model-id", required=True)
    export.add_argument("--checkpoint")
    export.add_argument("--downstream-checkpoint")
    export.add_argument("--registry")
    export.add_argument("--sam2-root", required=True)
    export.add_argument("--distill-root")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--device", default="cuda")
    export.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    export.add_argument("--reuse-downstream-dir")

    build = subparsers.add_parser("build", help="build TensorRT engines on Thor")
    build.add_argument("--bundle-dir", required=True)
    build.add_argument("--precision", choices=("fp32", "tf32", "fp16", "bf16"), required=True)
    build.add_argument("--workspace-gib", type=float, default=8.0)
    build.add_argument("--builder-optimization-level", type=int, choices=range(6), default=5)
    build.add_argument("--max-aux-streams", type=int, default=0)
    build.add_argument("--allow-non-thor", action="store_true", help=argparse.SUPPRESS)

    validate = subparsers.add_parser("validate", help="apply the no-accuracy-loss gate")
    validate.add_argument("--baseline", required=True)
    validate.add_argument("--candidate", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--maximum-metric-drop", type=float, default=0.1)
    validate.add_argument("--minimum-frame-iou", type=float, default=0.999)
    validate.add_argument("--bundle-dir")

    benchmark = subparsers.add_parser("benchmark", help="summarize a runtime JSONL trace")
    benchmark.add_argument("--trace", required=True)
    benchmark.add_argument("--output", required=True)

    engine_benchmark = subparsers.add_parser(
        "benchmark-engine", help="microbenchmark one TensorRT engine on CUDA"
    )
    engine_benchmark.add_argument("--engine", required=True)
    engine_benchmark.add_argument(
        "--role",
        choices=("encoder", "prompt_point_step", "prompt_box_step", "track_step"),
        required=True,
    )
    engine_benchmark.add_argument("--batch", type=int, choices=(1, 2, 4, 8), default=1)
    engine_benchmark.add_argument("--warmup", type=int, default=20)
    engine_benchmark.add_argument("--runs", type=int, default=100)
    engine_benchmark.add_argument("--output", required=True)

    pytorch_graphs = subparsers.add_parser(
        "benchmark-pytorch-graphs", help="microbenchmark the matching PyTorch export graphs"
    )
    pytorch_graphs.add_argument("--model-id", required=True)
    pytorch_graphs.add_argument("--checkpoint")
    pytorch_graphs.add_argument("--registry")
    pytorch_graphs.add_argument("--sam2-root", required=True)
    pytorch_graphs.add_argument("--batch", type=int, choices=(1, 2, 4), default=1)
    pytorch_graphs.add_argument("--warmup", type=int, default=20)
    pytorch_graphs.add_argument("--runs", type=int, default=100)
    pytorch_graphs.add_argument("--tf32", action="store_true")
    pytorch_graphs.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-bundle", help="verify checkpoint and engine hashes")
    verify.add_argument("--bundle-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-models":
        print("\n".join(sorted(load_registry(args.registry)["models"])))
        return 0
    if args.command == "probe":
        report = write_probe(args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "pin":
        print(json.dumps(pin_environment(args.probe, args.output), indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        spec = resolve_model(
            args.model_id,
            registry_path=args.registry,
            checkpoint=args.checkpoint,
            downstream_checkpoint=args.downstream_checkpoint,
        )
        export_bundle(
            spec,
            args.output_dir,
            sam2_root=args.sam2_root,
            distill_root=args.distill_root,
            device=args.device,
            dtype=args.dtype,
            reuse_downstream_dir=args.reuse_downstream_dir,
        )
        return 0
    if args.command == "build":
        build_bundle(
            args.bundle_dir,
            precision=args.precision,
            workspace_gib=args.workspace_gib,
            allow_non_thor=args.allow_non_thor,
            builder_optimization_level=args.builder_optimization_level,
            max_aux_streams=args.max_aux_streams,
        )
        return 0
    if args.command == "validate":
        result = accuracy_gate(
            args.baseline,
            args.candidate,
            maximum_metric_drop=args.maximum_metric_drop,
            minimum_frame_iou=args.minimum_frame_iou,
        )
        write_gate_result(result, args.output)
        if args.bundle_dir:
            manifest_path = Path(args.bundle_dir) / "manifest.json"
            manifest = BundleManifest.read(manifest_path)
            manifest.accuracy = result.to_dict()
            manifest.accuracy["baseline_report"] = str(Path(args.baseline).resolve())
            manifest.accuracy["candidate_report"] = str(Path(args.candidate).resolve())
            manifest.write(manifest_path)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.passed else 2
    if args.command == "benchmark":
        summary = summarize_trace(args.trace)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-engine":
        result = benchmark_engine(
            args.engine,
            role=args.role,
            batch=args.batch,
            warmup=args.warmup,
            runs=args.runs,
        )
        write_engine_benchmark(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-pytorch-graphs":
        spec = resolve_model(
            args.model_id,
            registry_path=args.registry,
            checkpoint=args.checkpoint,
        )
        result = benchmark_pytorch_graphs(
            spec,
            sam2_root=args.sam2_root,
            batch=args.batch,
            warmup=args.warmup,
            runs=args.runs,
            allow_tf32=args.tf32,
        )
        write_pytorch_graph_benchmark(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-bundle":
        root = Path(args.bundle_dir)
        errors = BundleManifest.read(root / "manifest.json").verify_files(root)
        if errors:
            print("\n".join(errors))
            return 2
        print("bundle hashes verified")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
