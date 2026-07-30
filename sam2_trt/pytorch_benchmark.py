from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from .export import _load_official_model, _modules
from .model_registry import ModelSpec
from .real_rope import patch_real_rope


def _timing_summary(timings: list[float]) -> dict[str, float]:
    values = np.asarray(timings, dtype=np.float64)
    average = mean(timings)
    return {
        "mean_ms": average,
        "median_ms": median(timings),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "enqueue_fps": 1000.0 / average,
    }


def _measure(torch, module, inputs, stream, warmup: int, runs: int) -> dict[str, float]:
    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(warmup):
            module(*inputs)
        stream.synchronize()
        timings = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            module(*inputs)
            end.record(stream)
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return _timing_summary(timings)


def benchmark_pytorch_graphs(
    spec: ModelSpec,
    *,
    sam2_root: str | Path,
    batch: int = 1,
    warmup: int = 20,
    runs: int = 100,
    allow_tf32: bool = False,
) -> dict[str, object]:
    import torch

    if spec.encoder != "sam2":
        raise ValueError("the PyTorch graph benchmark currently supports official SAM2 encoders")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PyTorch graph benchmarking")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    model = _load_official_model(spec, sam2_root, "cuda")
    patch_real_rope(model)
    stream = torch.cuda.Stream()
    dtype = torch.float32

    with torch.inference_mode(), torch.cuda.stream(stream):
        image = torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=dtype)
        official = model.forward_image(image)
        _, _, positions, sizes = model._prepare_backbone_features(official)
        base_position = positions[-1].permute(1, 2, 0).reshape(1, 256, *sizes[-1])
        encoder, point, box, mask, track = _modules(
            torch, model, model, base_position
        )
        high0 = torch.zeros(batch, 32, 256, 256, device="cuda", dtype=dtype)
        high1 = torch.zeros(batch, 64, 128, 128, device="cuda", dtype=dtype)
        embedding = torch.zeros(batch, 256, 64, 64, device="cuda", dtype=dtype)
        point_coords = torch.zeros(batch, 1, 2, device="cuda", dtype=dtype)
        point_labels = torch.ones(batch, 1, device="cuda", dtype=torch.int32)
        box_coords = torch.zeros(batch, 2, 2, device="cuda", dtype=dtype)
        box_labels = torch.tensor([[2, 3]], device="cuda", dtype=torch.int32).expand(batch, -1)
        mask_input = torch.zeros(
            batch, 1, 1024, 1024, device="cuda", dtype=dtype
        )
        position = base_position.expand(batch, -1, -1, -1)
        memory = torch.zeros(4, 4096, batch, 64, device="cuda", dtype=dtype)
        memory_position = torch.zeros_like(memory)
        memory_temporal = torch.zeros(4, batch, device="cuda", dtype=torch.int64)
        pointers = torch.zeros(8, batch, 256, device="cuda", dtype=dtype)
        pointer_distance = torch.zeros(8, batch, device="cuda", dtype=torch.int64)
    stream.synchronize()

    workloads = {
        "encoder": (encoder, (image,)),
        "prompt_point_step": (point, (high0, high1, embedding, point_coords, point_labels)),
        "prompt_box_step": (box, (high0, high1, embedding, box_coords, box_labels)),
        "prompt_mask_step": (mask, (high0, high1, embedding, mask_input)),
        "track_step": (
            track,
            (
                high0,
                high1,
                embedding,
                position,
                memory,
                memory_position,
                memory_temporal,
                pointers,
                pointer_distance,
            ),
        ),
    }
    results = {
        role: _measure(torch, module, inputs, stream, warmup, runs)
        for role, (module, inputs) in workloads.items()
    }
    return {
        "model_id": spec.model_id,
        "checkpoint": str(spec.checkpoint),
        "batch": batch,
        "warmup": warmup,
        "runs": runs,
        "tf32": allow_tf32,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "graphs": results,
    }


def write_pytorch_graph_benchmark(result: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
