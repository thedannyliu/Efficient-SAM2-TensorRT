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
candidate uses batch 1 for tracking on Thor in `eb35090`. It still requires a
complete multi-object pipeline measurement before promotion.

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

## Three-encoder FP16 comparison

TV11M and TV21M export/build reused the three downstream graphs and engines
only after downstream checkpoint, ONNX, engine, precision, and Thor device
checks passed. All three bundles point to the same downstream engine inodes.

| Encoder | ONNX size | Engine build | Engine size | Mean / p90 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| TV5M | 12.5 MB | 392.305 s | 13.8 MB | 6.358 / 6.476 / 6.534 |
| TV11M | 18.4 MB | 419.297 s | 28.2 MB | 7.359 / 7.471 / 7.577 |
| TV21M | 80.3 MB | 408.382 s | 38.3 MB | 14.375 / 14.563 / 14.989 |

Using the shared 14.967 ms batch-1 track engine, the engine-only single-object
steady-state estimates are 21.32 ms (46.9 FPS), 22.33 ms (44.8 FPS), and
29.34 ms (34.1 FPS) for TV5M/11M/21M respectively.

| Encoder | Point mean / minimum IoU | Box mean / minimum IoU | FP16 status |
| --- | ---: | ---: | --- |
| TV5M | 0.9315 / 0.4736 | 0.9940 / 0.9810 | Point fails |
| TV11M | 0.9962 / 0.9911 | 0.9979 / 0.9917 | Pass |
| TV21M | 0.9908 / 0.9529 | 0.9962 / 0.9896 | Pass |

The first TV5M point candidate promoted only three standard
`torch.nn.LayerNorm` modules to FP32. It left seven `LayerNorm2d` ReduceMean
operations in FP16, retained TensorRT's overflow warning, and produced
2.377 ms point latency with mean/minimum IoU 0.9316/0.4719. It is rejected.
The next candidate also promotes the seven SAM2 `LayerNorm2d` Reduce/Pow
operations; ONNX inspection confirms zero FP16 ReduceMean nodes in the point
graph before engine build.

## Camera latency versus FPS check

Commit `eade8e8` separates three quantities that were easy to confuse. A
follow-up check found that capacity still used arrival-to-completion latency,
which includes queue wait. The corrected trace therefore records
`worker_total_ms` separately:

- `processing_capacity_fps`: reciprocal of the current frame's
  `worker_total_ms`, excluding queue wait;
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
| Capacity from mean callback (legacy run) | 24.53 FPS |
| Measured throughput | 1.98 FPS |
| Frame interval p50 / p90 / max | 200.11 / 1601.49 / 3069.04 ms |
| Mean source-to-result age | 100.53 ms |
| Queue drops | 0 |

Camera timestamp intervals and worker intervals correlated at `0.9999998`; the
mean absolute difference was 0.235 ms. The throughput loss is therefore
upstream camera/USB delivery, not TensorRT latency or latest-frame queue loss.
The driver reported USB 2.1 and warned about reduced performance. Repeat the
same run with the corrected worker metric and after the D455F negotiates USB 3
before recording final camera FPS. The correction has negligible effect on
this legacy run because its measured queue wait was negligible.

The corrected node was then rebuilt and run on Thor at commit `8fc2769`.
Across 37 frames, `callback_total_ms - queue_wait_ms - worker_total_ms` had a
maximum absolute error of 0.001 ms (the JSON rounding unit), and the summary
recomputed 36 intervals over 32.292 s as 1.115 FPS. The 16 one-object frames
had 42.303 ms mean worker time, or 23.64 FPS processing capacity; their
interval-based throughput was only 0.863 FPS. Queue wait averaged 0.096 ms and
no frame was overwritten in the node. `lsusb -t` still showed the D455F on
480 Mbit/s USB 2, while the driver requested 640x480 RGB8 at 15 FPS. The
observed intervals ranged from 66.5 ms to 10.21 s, confirming that the
remaining FPS deficit is irregular upstream delivery rather than TensorRT
service time or a metric formula error.

After moving the D455F to USB-A, Thor negotiated USB 3.2 at 5000 Mbit/s and
published 1280x720 RGB8 at a stable 29.99 FPS. The first interactive viewer
implementation exposed a separate display bug: `/sam/result_json` could
arrive before `/sam/object_masks`, causing a raw frame to be displayed before
the same-stamp mask and producing visible flashing. Repeated test clicks also
accumulated four objects, increasing TV5M inference to 117--126 ms and reducing
tracking to about 8 FPS. The corrected viewer waits for every object mask
listed by the same-stamp result and replaces the current object on each new
prompt by default. This is a visualization synchronization fix, not a
TensorRT precision or engine change.

The rebuilt viewer at `4e8ce78` was reset and given one box prompt. Over the
latest 200 one-object frames, TV5M measured 37.43 ms mean inference
(p50/p90/p99 37.03/39.54/42.30 ms), 38.04 ms mean worker time, and 21.30 FPS
interval throughput. Mean source age was 79.81 ms and five latest-frame
overwrites occurred. This residual latency can make motion less fluid than the
30 FPS camera, but the viewer no longer alternates a raw frame with its
same-stamp masked frame.

The active interactive test uses `sam2.1-tinyvit-5m/fp16_aux0`: the distilled
TinyViT-5M image encoder plus the SAM2.1-L prompt, mask, memory, and pointer
components, all exported as FP16 engines. The verified pure-engine means are
6.328 ms encoder, 2.405 ms point prompt, 2.543 ms box prompt, and 14.967 ms
batch-1 track. Therefore the steady one-object engine-only lower bound is
21.294 ms; the live 37.43 ms `inference_ms` also includes transfers,
preprocessing, state construction, allocations, mask conversion, and
synchronization. The viewer can be launched with `replace_on_prompt:=false`
to accumulate up to eight objects while retaining same-stamp mask commit.

## Same-checkpoint PyTorch versus TensorRT camera A/B

On 2026-07-24, the native ROS baseline and TensorRT node were run headless on
the same Thor, RealSense 1280x720 RGB8 30 FPS stream, queue depth 1, center
point prompt, and one object. The native node used a 32-frame retained-history
cap; both paths use SAM2's standard seven-slot memory selection. Each
statistic uses 100 steady tracking frames. Both native checkpoint names hash
to the exact encoder checkpoint used by the TensorRT bundle:

```text
cd442f19b67be084305ead07908a21a911d25c3980f5f67e4b568db4d88878cf
```

| Pipeline | Measured FPS | Model/pipeline mean (ms) | p50 / p90 / p99 (ms) | Source-to-result mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| Native PyTorch FP32 | 3.840 | 177.542 | 177.443 / 177.938 / 179.027 | 312.423 |
| TensorRT FP16 | 26.505 | 34.080 | 33.961 / 35.043 / 35.672 | 80.990 |

TensorRT therefore delivered `6.90x` the camera-pipeline throughput, a `590.2%`
FPS increase. Its measured model/pipeline latency was `5.21x` faster, or
`80.8%` lower. This latency comparison is conservative: native
`latency_ms` brackets the PyTorch online model step but excludes frame
preprocessing and publishing, while TensorRT `inference_ms` includes host to
device transfer, preprocessing, all engines, state/mask conversion, and
synchronization. Throughput is computed from the 100 processed source stamps,
not an average of instantaneous FPS.

The native rows are retained only on Thor under
`results/thor/ab_tv5_native/native_100.jsonl`; TensorRT rows are under
`results/thor/ab_tv5_trt/trt_100.jsonl`. They are generated results and remain
untracked.

## Viewer motion/FPS diagnosis and optimization

Commit `d25b2af` split the previously ambiguous viewer `output FPS` into
tracker, UI receive, and unique-overlay present rates, and added compose and
display timing. With the original NumPy per-pixel alpha blend, a one-object
viewer run averaged 21.50 ms just for mask composition and 4.92 ms for
display. This was enough to make the visible frame rate lower than the model
result rate even though TensorRT latency did not change with scene motion.

Commits `3b6f096` and `f1b2558` moved alpha blending into OpenCV and limited it
to the mask bounding rectangle:

| Viewer implementation | Mean compose (ms) | Mean display (ms) |
| --- | ---: | ---: |
| NumPy selected-pixel blend | 21.504 | 4.916 |
| OpenCV full-frame masked blend | 8.551 | 5.211 |
| OpenCV mask-region blend | 3.799 | 5.236 |

The final mask composition is `82.3%` lower than the original. It changes only
visualization; engine outputs and mask accuracy are unchanged. Contour drawing
is now optional and off by default.

The remaining display-on run measured 20.51 FPS and 37.61 ms mean TensorRT
inference over 200 frames, versus the 26.51 FPS and 34.08 ms headless run.
During the same period, `/camera/camera/color/image_raw` had rolling rates of
about 16.7--23.2 FPS with 134--300 ms maximum gaps. Therefore residual visible
stutter is a combination of irregular RealSense delivery and the CPU/memory
cost of copying, compositing, and presenting ROS images; it is not caused by
motion making the TensorRT graph substantially slower.

## Unified GI-to-SAM2 shared camera transport

The SAM3/SAM2 unified UI originally serialized each 1280x720 BGR8 frame as a
2.76 MB ROS message from a Python MJPEG adapter to this C++ node. The adapter
received about 30 FPS, but the SAM2 result stream completed only 15.506 FPS
with no objects. Commit `6f1c15c` adds an optional locked latest-frame reader
under `/dev/shm`; the normal ROS subscription remains available for frozen
prompt initialization. SAM3 repo commit `12dc408` supplies that buffer.

The no-object headless result increased to 29.210 FPS (+88.4%) with mean source
age 19.44 ms. SAM3 commit `8d85337` then uses OpenCV's optimized BGR-to-RGB
conversion before the shared write, and SAM2 commit `87a4350` consumes the
shared payload as RGB8. This removes the per-pixel C++ channel loop without
changing pixel values.

Final displayed one-object results at 1280x720@30, after 100 warm-up and 500
measured frames:

| Model | FPS | Inference | Source age |
|---|---:|---:|---:|
| TV5M | 27.218 | 34.112 ms | 48.303 ms |
| TV11M | 27.046 | 35.737 ms | 54.154 ms |
| TV21M | 24.908 | 39.050 ms | 58.135 ms |

The transport is optional: leaving `shared_memory_path` empty preserves the
original ROS-only camera input.

## Remaining end-to-end optimization roadmap

The measured single-object headless path is 34.08 ms, while its encoder and
track engines sum to 21.29 ms. The 12.79 ms difference is now the largest
no-accuracy-loss target. The current C++ runtime allocates and frees device
storage for the input, normalized tensor, every engine output, memory banks,
temporal inputs, and output mask on each frame. It also allocates a pageable
host mask, copies device-to-host, and synchronizes the only CUDA stream before
returning.

The following order separates measured facts from projected targets:

| Priority | Change | Evidence and acceptance target | Accuracy risk |
| ---: | --- | --- | --- |
| 0 | Run reproducible clocks | Thor was in 120 W mode, not MAXN, and `jetson_clocks` was not locked. Repeat every candidate under one fixed mode after the user runs the required `sudo` commands. | None |
| 1 | Add CUDA-event stage timing | Split H2D, preprocess, encoder, state pack, track, mask resize, D2H, allocation, and sync. Explain at least 90% of the 12.79 ms engine/live gap before larger refactors. | None |
| 2 | Persistent tensor arena and pinned double buffers | Reuse fixed-shape engine I/O and state-pack buffers; replace per-frame `cudaMalloc/cudaFree`; use pinned RGB/mask staging. Initial goal: 24--27 ms one-object `inference_ms`, p99 below 30 ms. | None |
| 3 | Overlap encoder and tracking | Encode frame `n+1` on a second stream while frame `n` runs the state-dependent track step; overlap mask D2H on a copy stream. The engine throughput floor changes from `6.33 + 14.97 = 21.29 ms` toward `max(6.33, 14.97) = 14.97 ms`, subject to Thor contention. | None |
| 4 | Remove local ROS copies and backpressure | Make the tracker a composable node with the RealSense component and intra-process communication. Publish masks with best-effort keep-last-1 QoS and skip legacy/full-resolution outputs with no subscribers. | None |
| 5 | Publish one small preview for the UI | Compose object colors in C++/CUDA and publish one 640x360 or 960x540 preview. The Python viewer should not receive a full RGB frame plus one 1280x720 mask per object just to display them. Target at least 25 present FPS on a stable 30 FPS camera. | None |
| 6 | Parallel batch-1 contexts for multiple objects | Batch-2/4 engines are already slower than serial batch-1. Instead, test independent batch-1 execution contexts and streams after the shared encoder, with a device-resident state bank and one combined preview. | None |
| 7 | Lower precision only in the remaining track bottleneck | Add explicit Q/DQ and calibration for selected track Conv/MatMul operations while keeping normalization, softmax, IoU/object-score heads, memory encoder, and pointer logic in FP16/FP32. Promote only if video J&F and mask agreement remain at least 0.95. | Medium |
| 8 | Temporal/reduced-resolution approximations | Reuse encoder features on alternate frames, warp masks between model updates, or retrain for a smaller input. These can improve visible FPS beyond the exact-model ceiling but require motion-stratified accuracy tests. | High |

At 1280x720, one RGB8 frame is 2.76 MB and one mono8 object mask is
0.92 MB. A 30 FPS viewer therefore receives about 110.6 MB/s before DDS and
Python copies for one object; every additional full-resolution mask adds
27.6 MB/s. The current mask publisher also uses reliable history instead of a
latest-only sensor policy. This explains why UI load can disturb camera
delivery even after mask composition itself is reduced to 3--4 ms.

For one object at a fixed 30 FPS camera, the observable headless FPS can only
rise from 26.51 to about 30 FPS (`1.13x`) because the camera becomes the cap.
The more important gains are lower p99/source age, no dropped frames, and
multi-object scaling. With the current serial batch-1 engines, engine-only
steady-state estimates are:

| Objects | Encoder + serial track (ms) | Engine-only FPS |
| ---: | ---: | ---: |
| 1 | 21.29 | 46.96 |
| 2 | 36.26 | 27.58 |
| 4 | 66.19 | 15.11 |
| 8 | 126.06 | 7.93 |

This makes buffer reuse plus independent batch-1 stream overlap the primary
multi-object experiment. Do not retry batch-2/4 scheduling, increase ROS queue
depth, or quantize the already-small TV5 encoder first: those choices are
contradicted by the current Thor measurements or increase source age.

Artifacts remain ignored under:

```text
results/thor/tv5_fp16_aux0/engines/
results/thor/tv5_fp16_aux0/prompt_parity/
results/thor/tv5_fp16_aux0/box_parity/
results/thor/tv5_fp16_aux0/realsense_usb2_box_metrics_v2/
results/thor/tv5_fp16_aux0/realsense_metric_fix_8fc2769/
results/thor/ab_tv5_native/
results/thor/ab_tv5_trt/
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

## 60 FPS RealSense capacity benchmark

The D455F connected to Thor does not support 60 FPS at 1280x720. The highest
available 60 FPS color profile is `848x480x60`; the driver confirmed USB 3.2
and opened that exact RGB8 profile. With no tracked objects the composed
camera/tracker process received 59.656 FPS, with a 16.766 ms mean frame
interval and zero dropped frames. This is the preferred capacity test because
the earlier 30 FPS profile hid any tracker capacity above 30 FPS.

The following runs use TV5M FP16, the same camera scene and prompts,
`track_concurrency=8`, intra-process RealSense delivery, latest-only input,
and the asynchronous 640x360 preview. `inference_ms` is the complete C++
tracker wall time, not only the TensorRT plan time.

Persistent engine-output buffers and token-major state storage reduced the
synchronous path by 0.5--1.1%. This is a valid exact-output optimization, but
it also establishes that allocation and state transposition are no longer the
dominant multi-object cost.

| Objects | Synchronous inference (ms) | Synchronous FPS | Source age (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 32.352 | 30.755 | 56.497 |
| 2 | 51.450 | 19.381 | 75.316 |
| 4 | 92.284 | 10.847 | 116.176 |
| 8 | 178.982 | 5.589 | 203.131 |

The concurrency sweep for eight objects rejected limiting execution to fewer
streams:

| `track_concurrency` | Inference (ms) | FPS |
| ---: | ---: | ---: |
| 1 | 205.654 | 4.862 |
| 2 | 183.874 | 5.433 |
| 4 | 180.621 | 5.534 |
| 8 | 179.859 | 5.555 |

Eight concurrent batch-1 contexts are therefore retained. They improve the
eight-object result by 12.5% over serial execution, although the gain is much
smaller than ideal scaling because the track graphs contend for the same GPU
compute and memory bandwidth.

## Cross-frame encoder/tracker overlap

Commit `1472b5c` adds a double-buffered pipeline. While the object-dependent
track step consumes frame `N`, the shared TinyViT encoder computes frame
`N+1`. The result and preview retain frame `N`'s original ROS timestamp, so
masks are not drawn on the wrong image. No model operation, weight, dtype,
threshold, or state-selection rule changes.

| Objects | Overlap inference (ms) | Overlap FPS | Latency reduction | FPS gain | Source age (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.080 | 36.752 | 16.3% | 19.5% | 78.361 |
| 2 | 46.733 | 21.326 | 9.2% | 10.0% | 117.573 |
| 4 | 87.822 | 11.389 | 4.8% | 5.0% | 199.399 |
| 8 | 171.230 | 5.841 | 4.3% | 4.5% | 367.472 |

This is a throughput/latency tradeoff rather than a universal default. It
reduces the time between completed results, but a result is emitted one
processed frame later. At one object that is a useful 5.27 ms model-path
reduction for about 21.9 ms additional source age. At eight objects, the
fixed one-processed-frame delay is about 171 ms, so latency-sensitive robotics
should set `pipeline_overlap:=false` even though maximum-throughput tests use
`true`.

Trace directories on Thor are ignored and remain local:

```text
results/thor/tv5_60fps_baseline/
results/thor/tv5_60fps_concurrency{1,2,4}/
results/thor/tv5_60fps_reuse_v1/
results/thor/tv5_60fps_pipeline_v1/
```

## Unified UI OpenGL scaling

The unified General Instinct-to-SAM2 UI publishes a 640x360 asynchronous
preview but initially opens a resizable 2560x1440 window. Thor's OpenCV build
has Qt5 OpenGL support. Enabling its OpenGL HighGUI path at SAM3 repository
commit `1cc8594` reduced viewer CPU from about 78% to 3--4% without changing
the camera pixels, TensorRT engines, mask computation, preview dimensions, or
prompt.

All rows below are one-object, 1280x720@30 displayed runs with 100 warm-up
outputs and 500 measured outputs:

| Model | General viewer inference/FPS/source age | OpenGL viewer inference/FPS/source age | Inference change | Source-age change |
| --- | --- | --- | ---: | ---: |
| TV5M | 34.112 ms / 27.218 / 48.303 ms | 29.462 ms / 29.296 / 39.184 ms | -13.6% | -18.9% |
| TV11M | 35.737 ms / 27.046 / 54.154 ms | 30.744 ms / 29.910 / 41.870 ms | -14.0% | -22.7% |
| TV21M | 39.050 ms / 24.908 / 58.135 ms | 34.723 ms / 27.877 / 52.477 ms | -11.1% | -9.7% |

The OpenGL TV5M trace splits its 29.462 ms model path into 7.278 ms encoder
GPU time and 22.051 ms tracking-tail GPU time. Shared input transport is
0.951 ms and host image staging is 0.182 ms. The common SAM2 tracking tail,
not ROS transport, remains the largest exact-output optimization target.

## Shared track engine with independent contexts

Commit `377dfb0` replaces eight separately deserialized copies of the same
`track_step.fp16.engine` with one TensorRT engine and eight independent
execution contexts and CUDA streams. Output buffers and temporal state remain
separate per object. This preserves the existing parallel batch-1 execution
while sharing immutable engine weights.

The change is latency-neutral for one object: the same OpenGL TV5M workload
measured 29.443 ms inference versus 29.462 ms before the change. TV11M
hot-switch initialization fell from 3523.9 to 3246.9 ms (-7.9%). A four-point
smoke completed 172 measured results with object IDs `[1,2,3,4]`, 84.054 ms
mean inference, and no context-address or TensorRT enqueue errors. That smoke
used the live unified UI and is a correctness/capacity result, not a
same-scene latency comparison with the older direct-RealSense table.

The optimization is retained for lower duplicate engine state and faster
model switching, but it is not counted as steady-state tracking speedup.
Ignored traces are under
`results/benchmarks/shared_context_377dfb0/` on Thor.

## Unified 60 FPS input capacity

The RealSense's supported 60 FPS color profile is 848x480@60; it does not
offer 1280x720@60. After the licensed vendor runtime's 93.7 s camera-profile
restart, capture stabilized at 59.74 FPS. The displayed OpenGL UI and
latest-only shared camera transport remained enabled.

| Encoder | Objects | Completed FPS | Mean inference | Mean source age |
| --- | ---: | ---: | ---: | ---: |
| TV5M | 1 | 33.267 | 29.797 ms | 41.686 ms |
| TV11M | 1 | 32.650 | 30.427 ms | 42.387 ms |
| TV21M | 1 | 28.872 | 34.445 ms | 46.501 ms |
| TV5M | 2 | 22.397 | 44.541 ms | 57.422 ms |
| TV5M | 4 | 11.949 | 83.565 ms | 96.054 ms |

This removes the 30 FPS observation ceiling: TV5M's measured one-object
processing rate is about 33 FPS, and TV11M is only 1.9% slower. TV21M is
13.2% slower than TV5M. More input frames do not automatically reduce
source-age because a new latest frame arrives while inference is active; the
one-object queue-wait was about 10--11 ms.

A same-profile poll-rate screen compared the original 240 Hz shared-header
poll with 1000 Hz. Source-age was 41.872 and 41.686 ms respectively (-0.4%),
while inference varied in the opposite direction. The higher poll rate is
rejected as noise. SAM3 commit `a60e1cf` keeps the launch argument but restores
240 Hz as the default.

## Unified background concurrency update

The current shared-memory/OpenGL pipeline was remeasured at 848x480@60 after
the shared track-engine change. SAM3 commit `65dc282` exposes cross-frame
encoder/tracker overlap in the unified launcher.

| Objects | Sync inference / FPS / source age | Overlap inference / FPS / source age | Decision |
| ---: | --- | --- | --- |
| 1 | 29.219 ms / 33.766 / 42.072 ms | 26.225 ms / 37.641 / 65.250 ms | optional throughput mode |
| 2 | 46.667 ms / 21.356 / 61.250 ms | 45.429 ms / 21.923 / 105.665 ms | reject for interactive use |
| 4 | 83.527 ms / 11.952 / 98.314 ms | 85.667 ms / 11.657 / 185.864 ms | reject |

The one-object completed rate improves 11.5%, but the intentionally delayed
output adds 23.2 ms source age. At four objects, encoder and tracking kernels
contend and throughput falls 2.5%. The standalone low-latency default remains
`pipeline_overlap:=false`.

SAM3 commit `fab0c9e` exposes the existing per-object context limit. In the
same unified pipeline, four-object concurrency 1/2/4/8 measured
10.287/11.493/11.952/11.952 FPS. For eight objects, three 150-output
repetitions averaged:

| `track_concurrency` | Inference | Completed FPS | Source age |
| ---: | ---: | ---: | ---: |
| 4 | 162.78 ms | 6.136 | 176.86 ms |
| 8 | 166.75 ms | 5.993 | 181.58 ms |

Four streams preserve all model operations and results while avoiding
excessive GPU contention. The unified SAM3-to-SAM2 launcher now defaults to
four. The older direct-RealSense result above used separately deserialized
track engines and favored eight; it should not be substituted for this
current shared-engine/unified-pipeline A/B. Ignored Thor traces are under
`results/benchmarks/camera60_{sync,overlap}_65dc282/` and
`results/benchmarks/camera60_concurrency_fab0c9e/`.

## Object-count route transition

Commit `f19ab9e` adds `pipeline_overlap_max_objects`. When the overlap master
switch is enabled, the node uses cross-frame overlap only while the active
object count is at or below this threshold. The unified SAM3-to-SAM2 launch
uses a threshold of one:

- one object: overlap for the measured 11.5% isolated throughput gain;
- two or more: synchronous tracking to avoid the large source-age penalty and
  four-object slowdown.

Changing routes clears only an encoded frame that belongs to the old schedule.
It does not reset prompts, object memories, or IDs. Live Thor validation added
object `1`, then a second point. The result stream changed from
`pipeline_overlap:true, objects:[1]` to
`pipeline_overlap:false, objects:[1,2]`; both tracks continued without a
restart. The selected route, configured master switch, threshold, and temporal
delay are recorded in every `/sam/result_json` row.

The current 848x480@60 displayed pipeline also routes presentation cadence
separately so desktop painting does not starve TensorRT. With TV5M and four
tracking contexts, the measured one/two/four-object model rates were
32.43/19.13/11.26 FPS. This display scheduling does not change TensorRT
precision, model operations, masks, or per-object state.

## TensorRT object-bucket experiment

The runtime bucket design was adapted from
`SAM2-Distillation-Pipeline/docs/deployment/sam2_tinyvit_multiobject_thor.md`.
The reference Python implementation groups synchronized per-object histories
into capacity-four dynamic batches. Commit `ff2b5bf` implements the matching
TensorRT scheduler boundary:

- stable object order and independent per-object memories are preserved;
- only objects with equal selected memory/pointer counts share a batch;
- capacities 1, 2, and 4 are selectable with `track_bucket_size`;
- sessions below `track_bucket_min_objects` retain the batch-1 path;
- batch rows are packed on CUDA and outputs are split back to the original IDs;
- every result row records the configured and active bucket route.

This path is opt-in because the existing Thor track engine strongly favors
parallel batch-1 contexts. Same-session TV5M FP16, 848x480@60, headless mode-2
results for four objects were:

| Bucket size | Tracker latency | Tracking FPS | Source age | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1 | 86.49 ms | 11.53 | 101.16 ms | baseline |
| 2 | 189.90 ms | 5.26 | 203.25 ms | reject |
| 4 | 198.33 ms | 5.04 | 213.93 ms | reject |

At eight objects, bucket 4 measured 394.83 ms, 2.53 FPS, and 409.57 ms source
age, versus the existing batch-1/concurrency-4 result of 162.78 ms, 6.14 FPS,
and 176.86 ms. The engine-only batch 1/2/4 means were
14.65/54.30/111.40 ms. Unlike the H100 PyTorch reference, object throughput
per engine call nearly halves when batch increases on Thor TensorRT.

The experiment also found that the builder optimized track profiles around
four memories and eight pointers while steady runtime reaches seven and
sixteen. Commit `08ae58d` adds `--track-opt-max-state`. Rebuilding only the
track engine took 1029 seconds and used up to 2.60 GB activation memory. It
did not rescue batch 2/4, but the batch-1 four-object pipeline improved:

| Candidate | Tracker latency | Tracking FPS | Source age |
| --- | ---: | ---: | ---: |
| Original tactics | 86.49 ms | 11.53 | 101.16 ms |
| Full-state-opt tactics | 82.82 ms | 12.03 | 94.92 ms |

This is a 4.2% latency reduction and 4.4% tracking-FPS increase without
changing ONNX, weights, or precision. It remains a candidate until
same-input mask agreement is recorded. Bucket size 1 remains the deployed
default.

## SAM 3.1 Object Multiplex applicability

The official SAM 3.1 release and source were reviewed at upstream commit
`46957e47805eaa273f4aa7bbbd25a88bca9108ce`. Object Multiplex is not only a
runtime batching policy. The released model adds learned components:

- a fixed-capacity multiplex controller, normally 16 objects per bucket;
- a `MultiplexMaskDecoder` with per-slot learned mask, IoU, and object-score
  tokens;
- a multiplex memory encoder whose mask input has per-object channels;
- 256-channel decoupled memory attention and a tri-head vision neck;
- a separate `sam3.1_multiplex.pt` checkpoint.

Meta reports up to 16 objects in one forward pass and about 7x speedup at 128
objects on one H100 relative to the original SAM 3 implementation:

- https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- https://ai.meta.com/blog/segment-anything-model-3/

Using the released SAM 3.1 checkpoint for inference is training-free for a
deployment dataset: no user fine-tuning is required. Transplanting its
multiplex decoder into this SAM2 TinyViT pipeline is **not** training-free.
The present SAM2 track graph uses 64-channel memory, 1024 input resolution,
SAM2 decoder weights, and independent object states. SAM 3.1 uses incompatible
feature, memory, decoder-token, and checkpoint contracts.

The no-training subset of the idea is already represented here: one shared
image encode, fixed object slots, persistent state/output buffers, parallel
batch-1 contexts, batched preview work, and reduced CPU/GPU synchronization.
Achieving true joint-object reasoning requires a separate trained downstream
model. The proposed research branch is:

1. benchmark the official SAM 3.1 checkpoint on PACE H100 for
   1/2/4/8/16-object propagation;
2. export a fixed 16-slot propagation graph and audit ONNX/TensorRT support;
3. keep the TinyViT encoder frozen, add a compatible tri-neck/256-channel
   memory adapter, and distill the SAM 3.1 multiplex downstream path;
4. promote it only if video J&F and per-frame/object mask agreement remain at
   least 0.95 and Thor improves both throughput and memory use.
