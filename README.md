# SAM2 TensorRT for Jetson Thor

This repository is the deployment path for SAM2.1 Hiera Tiny/Small/Base+/Large and the
distilled TinyViT 21M/11M/5M encoders. It keeps the existing
`efficientsam3-benchmark` repository as the FP32 PyTorch oracle and the
`SAM2-Distillation-Pipeline` repository as the TinyViT implementation source.

The runtime is split into four strongly typed TensorRT graphs:

1. `encoder`: RGB-normalized 1024×1024 image to two high-resolution features, image
   embedding, and image position encoding.
2. `prompt_point_step` / `prompt_box_step`: preserve the official initial-prompt
   multimask policy while producing mask, score, object pointer, and memory.
3. `track_step`: exact selected memories and object pointers to the next mask and state.

The three TinyViT models reuse the SAM2.1-L prompt/track graphs. Memory attention is not
padded: objects are bucketed by actual memory/pointer length, while temporal positions
remain per object. Object batches use TensorRT profiles 1/2/4/8 and ignore padded rows.

## What is implemented

- Seven-model registry with exact checkpoint-path resolution and SHA256 recording.
- ONNX export for encoder/prompt/track graphs in FP32, FP16, or BF16.
- Thor-only TensorRT builder using `enqueueV3`, strongly typed networks on TensorRT 10,
  four object-batch profiles, and reusable timing cache.
- Exact SAM2.1 forward memory and object-pointer frame selection in Python and C++.
- C++ CUDA preprocessing, TensorRT execution, memory packing, mask postprocessing, and
  online point/box tracking for up to eight objects.
- ROS 2 latest-frame subscriber (queue depth 1), add/reset services, per-object masks,
  compatibility mask topic, and result JSON.
- Accuracy gate: full SA-V J&F and image mIoU may each drop at most 0.1 percentage point;
  every saved frame/object binary mask must have IoU at least 0.999 against the FP32
  same-checkpoint PyTorch oracle.

TensorRT plans are intentionally not included. All four plans must be built from the exact
checkpoint on the target Thor.

## Local logic checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
g++ -std=c++20 cpp/tests/state_selection_test.cpp -Icpp/include -o /tmp/sam2_state_test
/tmp/sam2_state_test
```

PACE is suitable for these checks and PyTorch-oracle evaluation. Do not build plans,
run ROS/camera validation, or claim deployment performance there.

## Thor workflow

First probe the actual system; do not guess JetPack, CUDA, TensorRT, or ROS versions:

```bash
sam2-trt probe --output results/thor_probe.json
sam2-trt pin --probe results/thor_probe.json --output environment.lock.json
```

Set checkpoint paths explicitly. TinyViT also needs the SAM2.1-L downstream checkpoint:

```bash
export SAM2_HIERA_LARGE_CHECKPOINT=/data/checkpoints/sam2.1_hiera_large.pt
export SAM2_TINYVIT_21M_CHECKPOINT=/data/checkpoints/tv21m_mse_cos.pt
```

Export and build an accuracy-first FP32 bundle:

```bash
sam2-trt export \
  --model-id sam2.1-hiera-large \
  --sam2-root /opt/src/sam2 \
  --output-dir bundles/sam2.1-hiera-large/fp32 \
  --dtype fp32

sam2-trt build \
  --bundle-dir bundles/sam2.1-hiera-large/fp32 \
  --precision fp32
```

For TinyViT, reuse the matching dtype's SAM2.1-L downstream graphs:

```bash
sam2-trt export \
  --model-id sam2.1-tinyvit-21m \
  --sam2-root /opt/src/sam2 \
  --distill-root /opt/src/SAM2-Distillation-Pipeline \
  --reuse-downstream-dir bundles/sam2.1-hiera-large/fp32 \
  --output-dir bundles/sam2.1-tinyvit-21m/fp32 \
  --dtype fp32
sam2-trt build --bundle-dir bundles/sam2.1-tinyvit-21m/fp32 --precision fp32
```

Repeat export/build for `tf32` (FP32 graph), `fp16`, and `bf16`. Start from no-TF32 FP32,
then choose the fastest candidate that passes the accuracy gate. Mixed precision and
FP8/INT8 are deliberately not auto-enabled: only introduce layer-level precision
changes after Thor profiling and real-input calibration, then run the same gate.

See [docs/thor_runbook.md](docs/thor_runbook.md) for C++/ROS build, services, topics,
benchmark capture, and acceptance procedure.
