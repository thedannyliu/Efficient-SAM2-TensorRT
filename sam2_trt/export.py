from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .manifest import BundleManifest, git_revision
from .model_registry import ModelSpec


def _prepend_import_path(path: str | Path | None):
    if path is None:
        return contextlib.nullcontext()

    class ImportPath:
        def __enter__(self):
            sys.path.insert(0, os.fspath(Path(path).resolve()))

        def __exit__(self, *_):
            sys.path.remove(os.fspath(Path(path).resolve()))

    return ImportPath()


def _load_official_model(spec: ModelSpec, sam2_root: str | Path, device: str):
    with _prepend_import_path(sam2_root):
        from sam2.build_sam import build_sam2

        model = build_sam2(
            spec.config if spec.encoder == "sam2" else "configs/sam2.1/sam2.1_hiera_l.yaml",
            os.fspath(spec.downstream_checkpoint),
            device=device,
            mode="eval",
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_tinyvit(spec: ModelSpec, distill_root: str | Path, device: str, downstream):
    import torch

    with _prepend_import_path(distill_root):
        from sam2_distill.models.stage1_checkpoint import (
            extract_state_dict,
            infer_adapter_mode,
            infer_stage1_model_name,
            infer_student_family,
            load_task_non_image_state,
        )
        from sam2_distill.models.stage1_student import build_stage1_student

        payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        task_load_summary = load_task_non_image_state(downstream, payload)
        state = extract_state_dict(payload)
        fallback = {
            384: "tiny_vit_21m_512.dist_in22k_ft_in1k",
            256: "tiny_vit_11m_224.dist_in22k_ft_in1k",
            160: "tiny_vit_5m_224.dist_in22k_ft_in1k",
        }[spec.tinyvit_embed_dim]
        model_name = infer_stage1_model_name(payload, state, fallback)
        family = infer_student_family(payload, model_name)
        student = build_stage1_student(
            student_family=family,
            model_name=model_name,
            checkpoint_path=None,
            adapter_mode=infer_adapter_mode(payload, state),
        )
        incompatible = student.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "distilled encoder checkpoint is not an exact match: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    student.to(device).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, task_load_summary


def _modules(torch, downstream, encoder, image_position):
    class OfficialEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            output = self.model.forward_image(image)
            _, features, positions, sizes = self.model._prepare_backbone_features(output)
            high0 = features[0].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[0])
            high1 = features[1].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[1])
            embedding = features[-1].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[-1])
            position = positions[-1].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[-1])
            return high0, high1, embedding, position

    class TinyViTEncoder(torch.nn.Module):
        def __init__(self, student, position):
            super().__init__()
            self.student = student
            self.register_buffer("position", position, persistent=True)

        def forward(self, image):
            output = self.student(image)
            position = self.position.expand(image.shape[0], -1, -1, -1)
            return output["high_res_s0"], output["high_res_s1"], output["image_embed"], position

    class PromptStep(torch.nn.Module):
        def __init__(self, model, multimask_output):
            super().__init__()
            self.model = model
            self.multimask_output = multimask_output

        def forward(self, high0, high1, embedding, point_coords, point_labels):
            batch = embedding.shape[0]
            pix = embedding.flatten(2).permute(2, 0, 1) + self.model.no_mem_embed
            pix = pix.permute(1, 2, 0).reshape(batch, 256, 64, 64)
            outputs = self.model._forward_sam_heads(
                backbone_features=pix,
                point_inputs={"point_coords": point_coords, "point_labels": point_labels},
                high_res_features=[high0, high1],
                multimask_output=self.multimask_output,
            )
            _, _, ious, _, high_masks, pointer, object_score = outputs
            memory, memory_position = self.model._encode_new_memory(
                current_vision_feats=[embedding.flatten(2).permute(2, 0, 1)],
                feat_sizes=[(64, 64)],
                pred_masks_high_res=high_masks,
                object_score_logits=object_score,
                is_mask_from_pts=True,
            )
            return high_masks, ious, pointer, object_score, memory, memory_position[-1]

    class TrackStep(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            high0,
            high1,
            embedding,
            position,
            mask_memory,
            mask_memory_position,
            mask_temporal_position,
            object_pointers,
            pointer_frame_distance,
        ):
            batch = embedding.shape[0]
            current = embedding.flatten(2).permute(2, 0, 1)
            current_position = position.flatten(2).permute(2, 0, 1)
            temporal_index = self.model.num_maskmem - mask_temporal_position - 1
            temporal = self.model.maskmem_tpos_enc[temporal_index].squeeze(2).squeeze(2).unsqueeze(1)
            mask_position = mask_memory_position + temporal
            mask_tokens = mask_memory.flatten(0, 1)
            mask_position_tokens = mask_position.flatten(0, 1)

            pointer_tokens = object_pointers.reshape(
                -1, batch, self.model.hidden_dim // self.model.mem_dim, self.model.mem_dim
            ).permute(0, 2, 1, 3).flatten(0, 1)
            if self.model.add_tpos_enc_to_obj_ptrs:
                position_dim = (
                    self.model.hidden_dim
                    if self.model.proj_tpos_enc_in_obj_ptrs
                    else self.model.mem_dim
                )
                half = position_dim // 2
                dim = torch.arange(half, dtype=torch.float32, device=embedding.device)
                denominator = 10000.0 ** (2 * (torch.div(dim, 2, rounding_mode="floor")) / half)
                normalized = pointer_frame_distance.float() / (self.model.max_obj_ptrs_in_encoder - 1)
                angle = normalized.unsqueeze(-1) / denominator
                pointer_position = torch.cat((angle.sin(), angle.cos()), dim=-1)
                pointer_position = pointer_position.to(dtype=object_pointers.dtype)
                pointer_position = self.model.obj_ptr_tpos_proj(pointer_position)
            else:
                pointer_position = object_pointers.new_zeros(
                    object_pointers.shape[0], batch, self.model.mem_dim
                )
            pointer_position = pointer_position.repeat_interleave(
                self.model.hidden_dim // self.model.mem_dim, dim=0
            )
            memory = torch.cat((mask_tokens, pointer_tokens), dim=0)
            memory_position = torch.cat((mask_position_tokens, pointer_position), dim=0)
            conditioned = self.model.memory_attention(
                curr=[current],
                curr_pos=[current_position],
                memory=memory,
                memory_pos=memory_position,
                num_obj_ptr_tokens=pointer_tokens.shape[0],
            )
            pix = conditioned.permute(1, 2, 0).reshape(batch, 256, 64, 64)
            outputs = self.model._forward_sam_heads(
                backbone_features=pix,
                point_inputs=None,
                high_res_features=[high0, high1],
                multimask_output=False,
            )
            _, _, ious, _, high_masks, pointer, object_score = outputs
            new_memory, new_position = self.model._encode_new_memory(
                current_vision_feats=[current],
                feat_sizes=[(64, 64)],
                pred_masks_high_res=high_masks,
                object_score_logits=object_score,
                is_mask_from_pts=False,
            )
            return high_masks, ious, pointer, object_score, new_memory, new_position[-1]

    encoder_module = OfficialEncoder(encoder) if encoder is downstream else TinyViTEncoder(encoder, image_position)
    return (
        encoder_module,
        PromptStep(downstream, True),
        PromptStep(downstream, False),
        TrackStep(downstream),
    )


def _export_one(torch, module, inputs, output: Path, input_names, output_names, dynamic_axes):
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        inputs,
        os.fspath(output),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=True,
        dynamic_axes=dynamic_axes,
        external_data=False,
        verify=False,
    )


def export_bundle(
    spec: ModelSpec,
    output_dir: str | Path,
    *,
    sam2_root: str | Path,
    distill_root: str | Path | None,
    device: str = "cuda",
    dtype: str = "fp32",
    reuse_downstream_dir: str | Path | None = None,
) -> Path:
    import torch

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    downstream = _load_official_model(spec, sam2_root, device)
    encoder = downstream
    task_load_summary = None
    if spec.encoder == "tinyvit":
        if distill_root is None:
            raise ValueError("--distill-root is required for TinyViT models")
        encoder, task_load_summary = _load_tinyvit(spec, distill_root, device, downstream)
        if task_load_summary is not None and reuse_downstream_dir:
            raise ValueError(
                "this distilled checkpoint contains task-tuned downstream weights; "
                "its prompt/track graphs cannot reuse the base SAM2.1-L graphs"
            )
    downstream.to(dtype=torch_dtype)
    encoder.to(dtype=torch_dtype)

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch_dtype)
        official = downstream.forward_image(dummy)
        _, _, positions, sizes = downstream._prepare_backbone_features(official)
        image_position = positions[-1].permute(1, 2, 0).reshape(1, 256, *sizes[-1])
        encoder_module, point_prompt_module, box_prompt_module, track_module = _modules(
            torch, downstream, encoder, image_position
        )
        encoder_module.eval()
        point_prompt_module.eval()
        box_prompt_module.eval()
        track_module.eval()

        common_outputs = [
            "mask_logits", "iou", "object_pointer", "object_score", "new_memory", "new_memory_position"
        ]
        _export_one(
            torch,
            encoder_module,
            (dummy,),
            output / "encoder.onnx",
            ["image"],
            ["high_res_s0", "high_res_s1", "image_embedding", "image_position"],
            {"image": {0: "batch"}, "high_res_s0": {0: "batch"}, "high_res_s1": {0: "batch"}, "image_embedding": {0: "batch"}, "image_position": {0: "batch"}},
        )

        high0 = torch.zeros(1, 32, 256, 256, device=device, dtype=torch_dtype)
        high1 = torch.zeros(1, 64, 128, 128, device=device, dtype=torch_dtype)
        embedding = torch.zeros(1, 256, 64, 64, device=device, dtype=torch_dtype)
        coords = torch.zeros(1, 2, 2, device=device, dtype=torch_dtype)
        labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        batch_outputs = {name: {0: "batch"} for name in common_outputs}
        if reuse_downstream_dir:
            source = Path(reuse_downstream_dir).resolve()
            for filename in ("prompt_point_step.onnx", "prompt_box_step.onnx", "track_step.onnx"):
                source_file = source / filename
                if not source_file.is_file():
                    raise FileNotFoundError(f"reused downstream graph is missing: {source_file}")
                destination = output / filename
                try:
                    os.link(source_file, destination)
                except OSError:
                    shutil.copy2(source_file, destination)
        else:
            for name, module, module_coords, module_labels in (
                ("prompt_point_step", point_prompt_module, coords[:, :1], torch.ones_like(labels[:, :1])),
                ("prompt_box_step", box_prompt_module, coords, labels),
            ):
                _export_one(
                    torch,
                    module,
                    (high0, high1, embedding, module_coords, module_labels),
                    output / f"{name}.onnx",
                    ["high_res_s0", "high_res_s1", "image_embedding", "point_coords", "point_labels"],
                    common_outputs,
                    {
                        "high_res_s0": {0: "batch"}, "high_res_s1": {0: "batch"},
                        "image_embedding": {0: "batch"}, "point_coords": {0: "batch"},
                        "point_labels": {0: "batch"}, **batch_outputs,
                    },
                )

            mask_memory = torch.zeros(1, 4096, 1, 64, device=device, dtype=torch_dtype)
            mask_pos = torch.zeros_like(mask_memory)
            mask_tpos = torch.zeros(1, 1, dtype=torch.int64, device=device)
            pointer = torch.zeros(1, 1, 256, device=device, dtype=torch_dtype)
            pointer_distance = torch.zeros(1, 1, dtype=torch.int64, device=device)
            _export_one(
                torch,
                track_module,
                (high0, high1, embedding, image_position, mask_memory, mask_pos, mask_tpos, pointer, pointer_distance),
                output / "track_step.onnx",
                ["high_res_s0", "high_res_s1", "image_embedding", "image_position", "mask_memory", "mask_memory_position", "mask_temporal_position", "object_pointers", "pointer_frame_distance"],
                common_outputs,
                {
                    "high_res_s0": {0: "batch"}, "high_res_s1": {0: "batch"},
                    "image_embedding": {0: "batch"}, "image_position": {0: "batch"},
                    "mask_memory": {0: "memory_frames", 2: "batch"},
                    "mask_memory_position": {0: "memory_frames", 2: "batch"},
                    "mask_temporal_position": {0: "memory_frames", 1: "batch"},
                    "object_pointers": {0: "pointer_frames", 1: "batch"},
                    "pointer_frame_distance": {0: "pointer_frames", 1: "batch"}, **batch_outputs,
                },
            )

    manifest = BundleManifest.create(
        model_id=spec.model_id,
        checkpoint=spec.checkpoint,
        downstream=spec.downstream,
        downstream_checkpoint=spec.downstream_checkpoint,
        source_revisions={
            "sam2": git_revision(sam2_root),
            "distillation": git_revision(distill_root) if distill_root else None,
            "sam2_trt_thor": git_revision(Path(__file__).resolve().parents[1]),
        },
    )
    manifest.environment["model_spec"] = asdict(spec) | {
        "checkpoint": os.fspath(spec.checkpoint),
        "downstream_checkpoint": os.fspath(spec.downstream_checkpoint),
    }
    manifest.environment["torch"] = torch.__version__
    manifest.environment["export_dtype"] = dtype
    manifest.environment["reused_downstream_dir"] = (
        os.fspath(Path(reuse_downstream_dir).resolve()) if reuse_downstream_dir else None
    )
    manifest.environment["task_model_load"] = task_load_summary
    manifest.write(output / "manifest.json")
    (output / "export.json").write_text(
        json.dumps(
            {"graphs": ["encoder.onnx", "prompt_point_step.onnx", "prompt_box_step.onnx", "track_step.onnx"]},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output
