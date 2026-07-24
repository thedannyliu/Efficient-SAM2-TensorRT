# Thor TensorRT optimization experiments

Last updated: 2026-07-23

## Goal and promotion rule

Optimize the distilled SAM2.1-L TinyViT 5M, 11M, and 21M pipelines on Jetson
AGX Thor, including the live RealSense ROS path. A candidate is promoted only
when it is the fastest measured end-to-end configuration whose binary masks
have mean IoU at least `0.95` against the original same-checkpoint PyTorch
model on identical frames and prompts. Minimum frame/object IoU is reported
separately. Dataset mIoU or J&F is reported only when ground-truth evaluation
is available and is not substituted by model-agreement IoU.

All source changes are made and tested on PACE, pushed to `origin/main`, and
then pulled on Thor. ONNX files, TensorRT engines, checkpoints, traces, and
camera outputs remain untracked Thor artifacts.

## Fixed inputs and provenance

| Item | Value |
| --- | --- |
| Thor | NVIDIA Jetson AGX Thor Developer Kit |
| L4T | R38.4 |
| Power mode | 120 W |
| TensorRT | 10.13.3.9 |
| PyTorch | 2.9.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| Compute capability | 11.0 |
| ROS | Jazzy |
| TensorRT repo starting commit | `5e37647720ec7bd1e39da1383ef19cc35a26616e` |
| SAM2 source commit | `2b90b9f` |
| Distillation source commit | `11dc3d38310020b6d9c64fc526f3bd966aaba1ee` |
| SAM2.1-L checkpoint SHA256 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| TinyViT-5M SHA256 | `cd442f19b67be084305ead07908a21a911d25c3980f5f67e4b568db4d88878cf` |
| TinyViT-11M SHA256 | `62a467bf915f4cf6b3c142743b55d0d4564d09658ee52896a69bcbc0fe5c77ab` |
| TinyViT-21M SHA256 | `da3a192cfd66aab4ed75fc9c3f804c84e3488540a905728ec2f319f5ab7a29fe` |
| Recorded stream inputs | `videos/test1.mov`, `videos/test2.mov` |
| Physical camera | Intel RealSense D455F |

The D455F was initially connected at `480M` (USB 2). Functional camera results
from that connection are valid, but camera throughput is not a final USB 3
performance result. The negotiated link, resolution, and measured publish FPS
must be recorded for every camera run.

## Thor artifact layout

```text
~/Efficient-SAM2-TensorRT/
  bundles/<model>/<candidate>/       # ONNX, engines, manifest; ignored
  logs/thor/                         # exporter/build/runtime logs; ignored
  results/thor/<run>/                # probes, benchmarks, traces; ignored

~/EfficientSAM3-Benchmark/
  checkpoints/sam2/sam2.1_hiera_large.pt
  checkpoints/distill/tv5.pt
  checkpoints/distill/tv11.pt
  checkpoints/distill/tv21.pt
  external/sam2/
  external/SAM2-Distillation-Pipeline/
```

## Bring-up history

| Stage | Result | Action |
| --- | --- | --- |
| Thor probe | CUDA, TensorRT, ROS, and PyTorch available | Saved under `results/thor/bringup_20260723/`. |
| Python tests | 21/21 passed at initial checkout | Confirmed package logic on Thor. |
| TinyViT checkpoint sync | All three SHA256 values match PACE | Use short names `tv5.pt`, `tv11.pt`, `tv21.pt`. |
| First 5M FP16 export | Failed before export because distillation checkout lacked `infer_adapter_mode` | Fast-forwarded the clean Thor dependency clone to pushed commit `11dc3d3`. |
| Second 5M FP16 export | Prompt SDPA received FP32 query and FP16 key/value | Added exporter-only FP16 autocast in `f9f6f21`. |
| Third 5M FP16 export | PyTorch 2.9 ONNX type-promotion bug on stability `sum(bool)` | Replaced count ratio with the mathematically equivalent mean ratio in exporter patch `c57201b`; exact formula unit test passes. |
| Track dynamic export | `memory_frames=1` violated a view-stride guard in expanded RoPE frequencies | Replaced expand/flatten with direct token-axis repeat in `128f47b`; complex-reference tests pass. |
| Complete 5M FP16 ONNX export | Four graphs passed `onnx.checker`; 47.998 s | Encoder 12.5 MB, point 13.9 MB, box 13.9 MB, track 60.4 MB. |
| C++ FP16 state handoff audit | Prompt graph emits an FP32 object pointer while the track graph accepts FP16 | Added explicit CUDA FP32-to-engine-dtype conversion in `68edbf2`; also removed batch-1 feature replication copies. |
| Complete 5M FP16 engine build | Four engine hashes passed `verify-bundle` | Track build took 312.337 s and produced a 152.7 MB engine. |
| ROS Jazzy build | Generic service callbacks and missing ament build metadata failed package discovery | Fixed in `e6f6ab5` and `abed0c0`; both packages then built and were visible to `ros2 pkg`. |
| Real-frame prompt parity | Box mean/minimum IoU 0.9940/0.9810; point mean/minimum 0.9315/0.4736 | Encoder FP16 is stable. One small-mask point sample changed the multimask selection, so all-FP16 is not yet promoted for every prompt type. |
| RealSense C++/ROS smoke | AddObject, TensorRT tracking, mask/result publish, trace, and process cleanup passed | Physical camera was limited by its USB 2.1 connection; see the measured distinction between capacity and throughput below. |

## Reproduction: TinyViT-5M FP16 baseline

```bash
cd "$HOME/EfficientSAM3-Benchmark"
source scripts/source_thor_ros_env.sh

export SAM2_HIERA_LARGE_CHECKPOINT="$PWD/checkpoints/sam2/sam2.1_hiera_large.pt"
export SAM2_TINYVIT_5M_CHECKPOINT="$PWD/checkpoints/distill/tv5.pt"

cd "$HOME/Efficient-SAM2-TensorRT"
sam2-trt export \
  --model-id sam2.1-tinyvit-5m \
  --sam2-root "$HOME/EfficientSAM3-Benchmark/external/sam2" \
  --distill-root "$HOME/EfficientSAM3-Benchmark/external/SAM2-Distillation-Pipeline" \
  --output-dir bundles/sam2.1-tinyvit-5m/fp16_aux0 \
  --dtype fp16

sam2-trt build \
  --bundle-dir bundles/sam2.1-tinyvit-5m/fp16_aux0 \
  --precision fp16 \
  --workspace-gib 8 \
  --builder-optimization-level 5 \
  --max-aux-streams 0
```

The first Thor encoder engine build used the compiler backend and took
`392.305 s`. TensorRT warned that FP16 LayerNorm Reduce/Pow after
self-attention can overflow. The all-FP16 result is therefore the speed
baseline; an FP32 LayerNorm/Reduce/Pow candidate must be evaluated for mask
agreement and latency before promotion.

Initial full-profile engine build progress:

| Graph | Build time | Engine size | Profiles |
| --- | ---: | ---: | --- |
| Encoder | 392.305 s | 13.8 MB | batch 1 |
| Point prompt | 802.662 s | 253.6 MB | batch 1/2/4/8 |
| Box prompt | 342.610 s | 253.6 MB | batch 1/2/4/8 |
| Track | 312.337 s | 152.7 MB | batch 1/2/4 |

## TinyViT-5M FP16 Thor measurements

Engine-only measurements used 20 warmups and 100 timed runs:

| Graph | Batch | Mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoder | 1 | 6.328 | 6.351 | 6.435 | 6.486 |
| Point prompt | 1 | 2.405 | 2.387 | 2.453 | 2.475 |
| Box prompt | 1 | 2.543 | 2.522 | 2.598 | 2.618 |
| Track | 1 | 14.967 | 15.122 | 15.205 | 15.269 |
| Track | 2 | 55.212 | 55.188 | 55.535 | 56.080 |
| Track | 4 | 113.317 | 113.302 | 113.916 | 114.221 |

The single-object steady-state engine sum is about `21.294 ms`, or 46.96 FPS
before preprocessing, state packing, mask transfer, ROS, and camera delivery.
Track batch 2/4 is slower than two/four batch-1 calls, so the next scheduler
candidate should use batch 1 for tracking on Thor and measure the complete
multi-object pipeline before promotion.

Prompt parity used `videos/test1.mov` and `videos/test2.mov`, frames
0/15/30/45, identical normalized prompts, threshold zero, and the same
checkpoint:

| Prompt | Mean mask IoU | Minimum mask IoU | Result |
| --- | ---: | ---: | --- |
| Box `(0.20, 0.20, 0.58, 0.88)` | 0.9940 | 0.9810 | Passes 0.95 rule |
| Point `(0.40, 0.55)` | 0.9315 | 0.4736 | Fails 0.95 rule |

Seven of eight point samples scored 0.9811--0.9998. The outlier reference and
candidate masks contained 2,293 and 4,151 foreground pixels. This localizes the
next precision experiment to point multimask scoring/selection and decoder
normalization instead of reverting the TinyViT encoder to FP32.

## Camera latency versus FPS check

Commit `eade8e8` separates three quantities that were easy to confuse:

- `processing_capacity_fps`: reciprocal of the current frame's callback
  latency;
- `processed_fps`: reciprocal of the interval between processed frame starts;
- summary `throughput_fps`: interval count divided by total measured interval
  duration.

Do not use the arithmetic mean of instantaneous `processed_fps` as run
throughput. A 640x480 RGB8, nominal 15 FPS, one-object box run produced:

| Metric | Result |
| --- | ---: |
| Object frames | 42 |
| Mean / p50 / p90 / p99 inference | 40.47 / 37.40 / 49.62 / 79.79 ms |
| Mean / p50 / p90 / p99 callback | 40.77 / 37.75 / 49.94 / 80.14 ms |
| Capacity from mean callback | 24.53 FPS |
| Measured throughput | 1.98 FPS |
| Frame interval p50 / p90 / max | 200.11 / 1601.49 / 3069.04 ms |
| Mean source-to-result age | 100.53 ms |
| Queue drops | 0 |

Camera timestamp intervals and worker intervals correlated at `0.9999998`; the
mean absolute difference was 0.235 ms. The throughput loss is therefore
upstream camera/USB delivery, not TensorRT latency or latest-frame queue loss.
The driver reported USB 2.1 and warned about reduced performance. Repeat the
same run after the D455F negotiates USB 3 before recording final camera FPS.

Artifacts remain ignored under:

```text
results/thor/tv5_fp16_aux0/engines/
results/thor/tv5_fp16_aux0/prompt_parity/
results/thor/tv5_fp16_aux0/box_parity/
results/thor/tv5_fp16_aux0/realsense_usb2_box_metrics_v2/
```

## Planned precision ablations

Run the same matrix independently for 5M, 11M, and 21M:

| Candidate | Encoder Conv/MatMul | Attention score | LayerNorm/Reduce/Pow | Prompt encoder | Mask decoder | Memory attention/encoder |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 oracle | FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
| FP16 baseline | FP16 | FP16 | FP16 | autocast FP16 | autocast FP16 | autocast FP16 |
| Safe normalization | FP16 | FP16 | FP32 | FP32 or autocast FP16 | autocast FP16 | autocast FP16 |
| High-precision prompt | FP16 | FP16 | selected | FP32 | selected | selected |
| Thor lower precision | calibrated FP8 where supported | FP16/FP32 | FP32 | FP16/FP32 | FP16 | FP16 |

Each row records engine-only latency by graph and batch, model-agreement mask
IoU, minimum frame/object IoU, and full camera pipeline latency/FPS. Auxiliary
stream limits `0`, `1`, and `2` are compared only after a precision candidate
passes the accuracy threshold.

## Camera optimization checklist

- RealSense actual USB link, selected profile, resolution, and measured ROS FPS.
- Best-effort image QoS and input queue depth 1.
- CUDA preprocessing directly into the encoder input dtype.
- No host copy between encoder features and prompt/track engines.
- Device-resident memory/object-pointer banks.
- GPU mask threshold/resize, transferring only the published mono8 mask.
- Prompt batches 1/2/4/8 and track batches 1/2/4 with 5–8 object splitting.
- `camera timestamp -> mask publish` mean/p50/p90/p99 latency.
- Processed FPS, source FPS, dropped frames, power mode, and `tegrastats`.

Physical-camera acceptance is performed with the TensorRT C++/ROS node, not
the existing PyTorch `sam2_online_tracking_node`.
