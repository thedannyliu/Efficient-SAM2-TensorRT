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
from .prompt_parity import _Engine, _preprocess
from .real_rope import patch_real_rope
from .validate import binary_iou


def sample_names(payload: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    images = {
        name.removeprefix("image__")
        for name in payload.files
        if name.startswith("image__")
    }
    masks = {
        name.removeprefix("mask__")
        for name in payload.files
        if name.startswith("mask__")
    }
    names = tuple(sorted(images.intersection(masks)))
    if not names:
        raise ValueError("sample NPZ has no matching image__/mask__ pairs")
    return names


def _cosine(torch, reference, candidate) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            reference.float().flatten(), candidate.float().flatten(), dim=0
        ).item()
    )


def compare_mask_prompt(
    spec: ModelSpec,
    *,
    sam2_root: str | Path,
    distill_root: str | Path | None,
    bundle_dir: str | Path,
    precision: str,
    samples: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mask prompt parity")
    downstream = _load_official_model(spec, sam2_root, "cuda")
    encoder = downstream
    if spec.encoder == "tinyvit":
        if distill_root is None:
            raise ValueError("distill_root is required for TinyViT")
        encoder, _ = _load_tinyvit(spec, distill_root, "cuda", downstream)
    patch_real_rope(downstream)
    patch_onnx_stability_scores(torch, downstream)

    root = Path(bundle_dir)
    encoder_engine = _Engine(root / f"encoder.{precision}.engine")
    mask_engine = _Engine(root / f"prompt_mask_step.{precision}.engine")
    payload = np.load(samples)
    names = sample_names(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 1024, 1024, device="cuda")
        official = downstream.forward_image(dummy)
        _, _, positions, sizes = downstream._prepare_backbone_features(official)
        image_position = positions[-1].permute(1, 2, 0).reshape(
            1, 256, *sizes[-1]
        )
        encoder_module, _, _, mask_module, _ = _modules(
            torch, downstream, encoder, image_position
        )
        encoder_module.eval()
        mask_module.eval()

        scores = {}
        masks = {}
        for name in names:
            image = np.asarray(payload[f"image__{name}"], dtype=np.uint8)
            mask = np.asarray(payload[f"mask__{name}"], dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{name} image must be RGB HWC")
            if mask.shape != image.shape[:2]:
                raise ValueError(f"{name} mask dimensions do not match image")
            model_image = _preprocess(torch, image)
            model_mask = torch.from_numpy(mask).to("cuda").reshape(
                1, 1, *mask.shape
            )
            model_mask = torch.nn.functional.interpolate(
                model_mask.float(),
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            ).ge_(0.5)

            reference_features = encoder_module(model_image)
            reference = mask_module(
                *reference_features[:3],
                model_mask.to(dtype=reference_features[0].dtype),
            )
            candidate_image = model_image.to(
                dtype=encoder_engine.input_dtype("image")
            ).contiguous()
            candidate_features = encoder_engine.run({"image": candidate_image})
            candidate = mask_engine.run(
                {
                    "high_res_s0": candidate_features["high_res_s0"],
                    "high_res_s1": candidate_features["high_res_s1"],
                    "image_embedding": candidate_features["image_embedding"],
                    "mask_input": model_mask.to(
                        dtype=mask_engine.input_dtype("mask_input")
                    ).contiguous(),
                }
            )
            torch.cuda.synchronize()

            reference_mask = reference[0][0, 0].gt(0).cpu().numpy()
            candidate_mask = candidate["mask_logits"][0, 0].gt(0).cpu().numpy()
            input_mask = model_mask[0, 0].cpu().numpy()
            masks[f"pytorch__{name}"] = reference_mask
            masks[f"tensorrt__{name}"] = candidate_mask
            scores[name] = {
                "mask_iou_pytorch": binary_iou(
                    reference_mask, candidate_mask
                ),
                "mask_iou_input": binary_iou(input_mask, candidate_mask),
                "object_pointer_cosine": _cosine(
                    torch, reference[2][0], candidate["object_pointer"][0]
                ),
                "new_memory_cosine": _cosine(
                    torch, reference[4][0], candidate["new_memory"][0]
                ),
                "new_memory_position_cosine": _cosine(
                    torch,
                    reference[5][0],
                    candidate["new_memory_position"][0],
                ),
            }

    np.savez_compressed(destination / "masks.npz", **masks)
    result = {
        "model_id": spec.model_id,
        "precision": precision,
        "samples": str(Path(samples).resolve()),
        "sample_count": len(names),
        "mean_mask_iou_pytorch": mean(
            item["mask_iou_pytorch"] for item in scores.values()
        ),
        "minimum_mask_iou_pytorch": min(
            item["mask_iou_pytorch"] for item in scores.values()
        ),
        "mean_mask_iou_input": mean(
            item["mask_iou_input"] for item in scores.values()
        ),
        "per_sample": scores,
    }
    (destination / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
