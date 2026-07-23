from __future__ import annotations

import json
import os
from pathlib import Path

from .manifest import BundleManifest, EngineRecord, sha256_file


def device_model() -> str:
    path = Path("/proc/device-tree/model")
    if not path.exists():
        return "unknown"
    return path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")


def require_thor(allow_non_thor: bool = False) -> None:
    model = device_model()
    if "thor" not in model.lower() and not allow_non_thor:
        raise RuntimeError(
            f"TensorRT engines must be built on Jetson Thor (detected {model!r}); "
            "use --allow-non-thor only for explicit development tests"
        )


def _shape_for(role: str, name: str, batch: int, endpoint: str):
    memory_frames = 1 if endpoint == "min" else 4 if endpoint == "opt" else 7
    pointer_frames = 1 if endpoint == "min" else 8 if endpoint == "opt" else 16
    prompts = 1 if role == "prompt_point_step" else 2
    shapes = {
        "image": (1, 3, 1024, 1024),
        "high_res_s0": (batch, 32, 256, 256),
        "high_res_s1": (batch, 64, 128, 128),
        "image_embedding": (batch, 256, 64, 64),
        "image_position": (batch, 256, 64, 64),
        "point_coords": (batch, prompts, 2),
        "point_labels": (batch, prompts),
        "mask_memory": (memory_frames, 4096, batch, 64),
        "mask_memory_position": (memory_frames, 4096, batch, 64),
        "mask_temporal_position": (memory_frames, batch),
        "object_pointers": (pointer_frames, batch, 256),
        "pointer_frame_distance": (pointer_frames, batch),
    }
    del role
    return shapes[name]


def _network_flags(trt):
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return flags


def _profile_batches(role: str) -> tuple[int, ...]:
    if role == "encoder":
        return (1,)
    if role == "track_step":
        return (1, 2, 4)
    return (1, 2, 4, 8)


def _validate_builder_options(builder_optimization_level: int, max_aux_streams: int) -> None:
    if builder_optimization_level not in range(6):
        raise ValueError("builder optimization level must be between 0 and 5")
    if max_aux_streams < 0:
        raise ValueError("max auxiliary streams must be non-negative")


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    role: str,
    workspace_gib: float = 8.0,
    allow_tf32: bool = False,
    timing_cache: str | Path | None = None,
    builder_optimization_level: int = 5,
    max_aux_streams: int = 0,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    _validate_builder_options(builder_optimization_level, max_aux_streams)
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(os.fspath(Path(onnx_path).resolve())):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    config.builder_optimization_level = builder_optimization_level
    config.max_aux_streams = max_aux_streams
    if not allow_tf32 and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    cache_path = Path(timing_cache) if timing_cache else None
    if cache_path:
        payload = cache_path.read_bytes() if cache_path.is_file() else b""
        cache = config.create_timing_cache(payload)
        config.set_timing_cache(cache, ignore_mismatch=False)

    for batch in _profile_batches(role):
        dynamic_inputs = [
            network.get_input(index)
            for index in range(network.num_inputs)
            if any(dimension == -1 for dimension in network.get_input(index).shape)
        ]
        if not dynamic_inputs:
            break
        profile = builder.create_optimization_profile()
        for tensor in dynamic_inputs:
            profile.set_shape(
                tensor.name,
                _shape_for(role, tensor.name, batch, "min"),
                _shape_for(role, tensor.name, batch, "opt"),
                _shape_for(role, tensor.name, batch, "max"),
            )
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {role} engine")
    destination = Path(engine_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(serialized)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    inputs = {
        network.get_input(index).name: list(network.get_input(index).shape)
        for index in range(network.num_inputs)
    }
    outputs = {
        network.get_output(index).name: list(network.get_output(index).shape)
        for index in range(network.num_outputs)
    }
    return inputs, outputs


def build_bundle(
    bundle_dir: str | Path,
    *,
    precision: str,
    workspace_gib: float = 8.0,
    allow_non_thor: bool = False,
    builder_optimization_level: int = 5,
    max_aux_streams: int = 0,
) -> Path:
    _validate_builder_options(builder_optimization_level, max_aux_streams)
    require_thor(allow_non_thor)
    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = BundleManifest.read(manifest_path)
    exported_dtype = manifest.environment.get("export_dtype")
    expected_dtype = "fp32" if precision in ("fp32", "tf32") else precision
    if exported_dtype != expected_dtype:
        raise ValueError(
            f"graph dtype is {exported_dtype}, but precision {precision} requires {expected_dtype} export"
        )
    records: list[EngineRecord] = []
    for role in ("encoder", "prompt_point_step", "prompt_box_step", "track_step"):
        engine_name = f"{role}.{precision}.engine"
        inputs, outputs = build_engine(
            root / f"{role}.onnx",
            root / engine_name,
            role=role,
            workspace_gib=workspace_gib,
            allow_tf32=precision == "tf32",
            timing_cache=root / "timing.cache",
            builder_optimization_level=builder_optimization_level,
            max_aux_streams=max_aux_streams,
        )
        records.append(
            EngineRecord(
                role=role,
                filename=engine_name,
                sha256=sha256_file(root / engine_name),
                precision=precision,
                inputs=inputs,
                outputs=outputs,
            )
        )
    manifest.engines = records
    manifest.environment["tensorrt_device_model"] = device_model()
    manifest.environment["builder_optimization_level"] = builder_optimization_level
    manifest.environment["max_aux_streams"] = max_aux_streams
    try:
        import tensorrt as trt

        manifest.environment["tensorrt"] = trt.__version__
    except ImportError:
        pass
    manifest.write(manifest_path)
    (root / "build.json").write_text(
        json.dumps(
            {
                "precision": precision,
                "builder_optimization_level": builder_optimization_level,
                "max_aux_streams": max_aux_streams,
                "engines": [record.filename for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
