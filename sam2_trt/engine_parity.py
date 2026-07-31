from __future__ import annotations

import json
from pathlib import Path

from .build import _shape_for
from .prompt_parity import _Engine


def compare_engines(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    role: str,
    batch: int,
    shape_endpoint: str = "max",
    seed: int = 0,
) -> dict[str, object]:
    import tensorrt as trt
    import torch

    if shape_endpoint not in {"min", "opt", "max"}:
        raise ValueError("shape endpoint must be min, opt, or max")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT parity")
    baseline = _Engine(baseline_path)
    candidate = _Engine(candidate_path)
    baseline_names = {
        baseline._engine.get_tensor_name(index)
        for index in range(baseline._engine.num_io_tensors)
    }
    candidate_names = {
        candidate._engine.get_tensor_name(index)
        for index in range(candidate._engine.num_io_tensors)
    }
    if baseline_names != candidate_names:
        raise ValueError("engine tensor names differ")

    torch.manual_seed(seed)
    inputs = {}
    for index in range(baseline._engine.num_io_tensors):
        name = baseline._engine.get_tensor_name(index)
        if (
            baseline._engine.get_tensor_mode(name)
            != trt.TensorIOMode.INPUT
        ):
            continue
        baseline_dtype = baseline.input_dtype(name)
        if candidate.input_dtype(name) != baseline_dtype:
            raise ValueError(f"engine input dtype differs for {name}")
        shape = (
            (1, 3, 1024, 1024)
            if role == "encoder"
            else _shape_for(role, name, batch, shape_endpoint)
        )
        if baseline_dtype.is_floating_point:
            inputs[name] = torch.randn(
                shape, dtype=baseline_dtype, device="cuda"
            )
        else:
            inputs[name] = torch.zeros(
                shape, dtype=baseline_dtype, device="cuda"
            )

    baseline_profile = (
        0
        if baseline._engine.num_optimization_profiles == 1
        or role == "encoder"
        else (1, 2, 4, 8).index(batch)
    )
    baseline_inputs = inputs
    if role == "track_shared_image_step":
        baseline_inputs = {
            name: (
                tensor.expand(batch, *tensor.shape[1:]).contiguous()
                if name
                in {
                    "high_res_s0",
                    "high_res_s1",
                    "image_embedding",
                    "image_position",
                }
                else tensor
            )
            for name, tensor in inputs.items()
        }
    baseline_outputs = baseline.run(
        baseline_inputs, profile=baseline_profile
    )
    candidate_outputs = candidate.run(inputs, profile=0)
    torch.cuda.synchronize()

    output_metrics = {}
    for name, reference in baseline_outputs.items():
        value = candidate_outputs[name]
        if reference.shape != value.shape:
            raise ValueError(f"engine output shape differs for {name}")
        reference_float = reference.float()
        value_float = value.float()
        delta = (reference_float - value_float).abs()
        denominator = reference_float.abs().clamp_min(1e-6)
        if reference_float.count_nonzero() == 0 and value_float.count_nonzero() == 0:
            cosine_value = 1.0
        else:
            cosine_value = float(
                torch.nn.functional.cosine_similarity(
                    reference_float.flatten(), value_float.flatten(), dim=0
                ).item()
            )
        item = {
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "max_relative": float((delta / denominator).max().item()),
            "cosine": cosine_value,
        }
        if name == "mask_logits":
            reference_mask = reference_float > 0
            value_mask = value_float > 0
            intersection = (reference_mask & value_mask).sum()
            union = (reference_mask | value_mask).sum()
            item["binary_iou"] = (
                1.0
                if union.item() == 0
                else float((intersection.float() / union).item())
            )
        output_metrics[name] = item

    return {
        "baseline_engine": str(Path(baseline_path).resolve()),
        "candidate_engine": str(Path(candidate_path).resolve()),
        "role": role,
        "batch": batch,
        "shape_endpoint": shape_endpoint,
        "seed": seed,
        "outputs": output_metrics,
    }


def write_engine_parity(
    result: dict[str, object], output_path: str | Path
) -> None:
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
