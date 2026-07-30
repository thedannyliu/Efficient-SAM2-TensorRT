from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

from .export import (
    _load_official_model,
    _load_tinyvit,
    _modules,
    patch_onnx_stability_scores,
)
from .model_registry import ModelSpec
from .real_rope import patch_real_rope
from .trt_benchmark import _torch_dtype
from .validate import binary_iou


class _Engine:
    def __init__(self, path: str | Path):
        import tensorrt as trt

        self._trt = trt
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self._context = self._engine.create_execution_context()

    def input_dtype(self, name: str):
        import torch

        return _torch_dtype(torch, self._trt, self._engine.get_tensor_dtype(name))

    def run(self, inputs: dict[str, object], *, profile: int = 0) -> dict[str, object]:
        import torch

        stream = torch.cuda.current_stream()
        if self._engine.num_optimization_profiles > 1:
            if not self._context.set_optimization_profile_async(profile, stream.cuda_stream):
                raise RuntimeError(f"failed to select optimization profile {profile}")
        for name, tensor in inputs.items():
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"TensorRT input {name} must be contiguous CUDA memory")
            if tensor.dtype != self.input_dtype(name):
                raise ValueError(
                    f"TensorRT input {name} has dtype {tensor.dtype}, "
                    f"expected {self.input_dtype(name)}"
                )
            if not self._context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"TensorRT rejected shape {tuple(tensor.shape)} for {name}")
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT input {name}")

        outputs = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_mode(name) != self._trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            tensor = torch.empty(
                shape,
                dtype=_torch_dtype(
                    torch, self._trt, self._engine.get_tensor_dtype(name)
                ),
                device="cuda",
            )
            outputs[name] = tensor
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT output {name}")
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return outputs


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame_index} from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _preprocess(torch, frame: np.ndarray):
    image = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda")
    image = image.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    image = torch.nn.functional.interpolate(
        image, size=(1024, 1024), mode="bilinear", align_corners=False
    )
    mean_tensor = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std_tensor = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return image.sub_(mean_tensor).div_(std_tensor)


def compare_prompt_masks(
    spec: ModelSpec,
    *,
    sam2_root: str | Path,
    distill_root: str | Path | None,
    bundle_dir: str | Path,
    precision: str,
    videos: list[str | Path],
    frame_indices: list[int],
    point: tuple[float, float] | None,
    box: tuple[float, float, float, float] | None,
    output_dir: str | Path,
) -> dict[str, object]:
    import torch

    if (point is None) == (box is None):
        raise ValueError("provide exactly one normalized point or box prompt")
    coordinates = point if point is not None else box
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError("normalized prompt coordinates must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prompt parity")

    downstream = _load_official_model(spec, sam2_root, "cuda")
    encoder = downstream
    if spec.encoder == "tinyvit":
        if distill_root is None:
            raise ValueError("distill_root is required for TinyViT")
        encoder, _ = _load_tinyvit(spec, distill_root, "cuda", downstream)
    patch_real_rope(downstream)
    patch_onnx_stability_scores(torch, downstream)

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 1024, 1024, device="cuda")
        official = downstream.forward_image(dummy)
        _, _, positions, sizes = downstream._prepare_backbone_features(official)
        image_position = positions[-1].permute(1, 2, 0).reshape(1, 256, *sizes[-1])
        encoder_module, point_module, box_module, _, _ = _modules(
            torch, downstream, encoder, image_position
        )
        prompt_module = point_module if point is not None else box_module
        encoder_module.eval()
        prompt_module.eval()

        root = Path(bundle_dir)
        encoder_engine = _Engine(root / f"encoder.{precision}.engine")
        role = "prompt_point_step" if point is not None else "prompt_box_step"
        prompt_engine = _Engine(root / f"{role}.{precision}.engine")
        labels = (1,) if point is not None else (2, 3)

        reference_masks = {}
        candidate_masks = {}
        scores = {}
        quality_scores = {}
        for video_value in videos:
            video = Path(video_value).resolve()
            for frame_index in frame_indices:
                frame = _read_frame(video, frame_index)
                image = _preprocess(torch, frame)
                coords = torch.tensor(coordinates, device="cuda").reshape(1, -1, 2)
                coords = coords * coords.new_tensor((1024.0, 1024.0))
                point_labels = torch.tensor(
                    labels, dtype=torch.int32, device="cuda"
                ).reshape(1, -1)

                reference_features = encoder_module(image)
                reference_outputs = prompt_module(
                    *reference_features[:3], coords, point_labels
                )
                reference = reference_outputs[0]

                candidate_image = image.to(
                    dtype=encoder_engine.input_dtype("image")
                ).contiguous()
                candidate_features = encoder_engine.run({"image": candidate_image})
                candidate_inputs = {
                    "high_res_s0": candidate_features["high_res_s0"],
                    "high_res_s1": candidate_features["high_res_s1"],
                    "image_embedding": candidate_features["image_embedding"],
                    "point_coords": coords.to(
                        dtype=prompt_engine.input_dtype("point_coords")
                    ).contiguous(),
                    "point_labels": point_labels,
                }
                candidate_outputs = prompt_engine.run(candidate_inputs)
                candidate = candidate_outputs["mask_logits"]
                torch.cuda.synchronize()

                key = f"{video.stem}_frame_{frame_index:06d}"
                reference_mask = reference[0, 0].gt(0).cpu().numpy()
                candidate_mask = candidate[0, 0].gt(0).cpu().numpy()
                reference_masks[key] = reference_mask
                candidate_masks[key] = candidate_mask
                scores[key] = binary_iou(reference_mask, candidate_mask)
                reference_iou = reference_outputs[1][0].float().cpu().tolist()
                candidate_iou = candidate_outputs["iou"][0].float().cpu().tolist()
                quality_scores[key] = {
                    "pytorch": reference_iou,
                    "pytorch_argmax": int(np.argmax(reference_iou)),
                    "tensorrt": candidate_iou,
                    "tensorrt_argmax": int(np.argmax(candidate_iou)),
                }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "pytorch_masks.npz", **reference_masks)
    np.savez_compressed(destination / "tensorrt_masks.npz", **candidate_masks)
    result = {
        "model_id": spec.model_id,
        "precision": precision,
        "videos": [str(Path(video).resolve()) for video in videos],
        "frame_indices": frame_indices,
        "normalized_prompt": {
            "kind": "point" if point is not None else "box",
            "coordinates": list(coordinates),
        },
        "mask_iou": {
            "mean": mean(scores.values()),
            "minimum": min(scores.values()),
            "per_frame": scores,
        },
        "samples": len(scores),
        "quality_scores": quality_scores,
        "reference_masks": "pytorch_masks.npz",
        "candidate_masks": "tensorrt_masks.npz",
    }
    (destination / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
