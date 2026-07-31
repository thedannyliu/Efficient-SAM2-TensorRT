from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from .build import _shape_for


def _torch_dtype(torch, trt, dtype):
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark dtype: {dtype}") from exc


def benchmark_engine(
    engine_path: str | Path,
    *,
    role: str,
    batch: int = 1,
    shape_endpoint: str = "opt",
    warmup: int = 20,
    runs: int = 100,
    cuda_graph: bool = False,
) -> dict[str, object]:
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT benchmarking")
    if shape_endpoint not in {"min", "opt", "max"}:
        raise ValueError("shape endpoint must be min, opt, or max")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    profile = (
        0
        if engine.num_optimization_profiles == 1 or role == "encoder"
        else (1, 2, 4, 8).index(batch)
    )
    if engine.num_optimization_profiles > 1:
        if not context.set_optimization_profile_async(profile, stream.cuda_stream):
            raise RuntimeError(f"failed to select optimization profile {profile}")

    tensors = {}
    inputs = []
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = (
                (1, 3, 1024, 1024)
                if role == "encoder"
                else _shape_for(role, name, batch, shape_endpoint)
            )
            if not context.set_input_shape(name, shape):
                raise RuntimeError(f"TensorRT rejected shape {shape} for {name}")
            inputs.append(name)
        else:
            outputs.append(name)
    with torch.cuda.stream(stream):
        for name in inputs + outputs:
            shape = tuple(context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"unresolved dynamic output shape for {name}: {shape}")
            tensor = torch.zeros(
                shape,
                dtype=_torch_dtype(torch, trt, engine.get_tensor_dtype(name)),
                device="cuda",
            )
            tensors[name] = tensor
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind {name}")

    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warmup enqueue failed")
    stream.synchronize()

    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT CUDA Graph capture enqueue failed")

    timings = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if graph is None:
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT benchmark enqueue failed")
        else:
            with torch.cuda.stream(stream):
                graph.replay()
        end.record(stream)
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    values = np.asarray(timings, dtype=np.float64)
    average = mean(timings)
    return {
        "engine": str(Path(engine_path).resolve()),
        "role": role,
        "batch": batch,
        "cuda_graph": cuda_graph,
        "shape_endpoint": shape_endpoint,
        "profile": profile,
        "warmup": warmup,
        "runs": runs,
        "mean_ms": average,
        "median_ms": median(timings),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "enqueue_fps": 1000.0 / average,
        "object_throughput_per_second": batch * 1000.0 / average,
        "inputs": {name: list(tensors[name].shape) for name in inputs},
        "outputs": {name: list(tensors[name].shape) for name in outputs},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tensorrt": trt.__version__,
    }


def write_engine_benchmark(result: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
