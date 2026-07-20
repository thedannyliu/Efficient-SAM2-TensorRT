# PACE TensorRT Experiment Log

Last updated: 2026-07-20

## Goal and acceptance criteria

The first target is a useful SAM2.1 Hiera Tiny TensorRT path before moving to
Jetson Thor. PACE is used only for ROS-free export, engine-build, correctness,
and kernel-latency checks. Final engine builds, camera-stream measurements, and
deployment acceptance remain Thor-only.

An optimized candidate is accepted only when all of the following hold:

- It is compared with the same checkpoint and prompt policy on the same GPU.
- End-to-end mask accuracy does not drop by more than 0.1 percentage point.
- Every saved candidate binary mask has IoU at least 0.999 against the FP32
  PyTorch same-checkpoint mask.
- Latency includes encoder, prompt or track tail, preprocessing, postprocessing,
  and camera transfer/synchronization where applicable.
- Mean, p50, p90, p99, throughput, GPU, software versions, and engine hashes are
  recorded. Engine files and generated results are intentionally ignored by Git.

## Current baseline

Job `11291342` ran the official SAM2.1 Hiera Tiny PyTorch implementation on one
L40S with QOS `embers`, checkpoint SHA256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`, and the
fixed ten-image COCO point-prompt manifest.

| Measurement | Result |
| --- | ---: |
| Mean total pipeline | 26.0901 ms |
| Effective pipeline rate | 38.3288 FPS |
| Mean predictor wall time | 24.8039 ms |
| Effective predictor rate | 40.3163 FPS |
| Mean best-mask IoU | 0.442887 |
| Repetitions | 50 rows: 10 images x 5 repeats |

This L40S result is the current baseline, but it is not a fair denominator for
H100 or RTX6000 TensorRT measurements. Same-GPU PyTorch baselines are required
before reporting acceleration percentages.

Job `11314933` completed the same image-predictor workload on H100:

| Measurement | Result |
| --- | ---: |
| Mean total pipeline | 20.1773 ms |
| Effective pipeline rate | 49.5607 FPS |
| Mean predictor wall time | 19.0847 ms |
| Effective predictor rate | 52.3980 FPS |
| Mean best-mask IoU | 0.442884 |

The TensorRT initial-point graph intentionally executes SAM2's three-mask
initial-click policy and selects the best candidate, while this image-predictor
baseline requests one mask directly. The H100 result is therefore a hardware
baseline, not yet the denominator for a final acceleration claim. An exact
same-policy PyTorch wrapper timing is required.

## Implementation progression

| Commit | Improvement | Evidence |
| --- | --- | --- |
| `541cb24` | Added model registry, four-graph ONNX/TensorRT bundle, validation gate, C++ CUDA runtime, and ROS 2 integration. | Python logic tests and C++ state-selection smoke. |
| `6f85f8e` | Added direct TensorRT `enqueueV3` engine microbenchmarking. | Emits per-role/batch JSON with mean and percentile latency. |
| `44954dc` | Replaced complex-number RoPE with equivalent real sin/cos pair rotation for export. | Matches the complex reference within `1e-6`. |
| `a125e35` | Made TensorRT 11 network creation use strongly typed mode without the removed explicit-batch flag. | TensorRT 10/11 flag unit tests. |
| `a8e4010` | Added a fixed batch-1 optimization profile when the encoder ONNX was dynamic. | Encoder engine built successfully. |
| `92bacf6` | Rewrote sizes-based 4-D ONNX Resize nodes to spatial scales so batch remains dynamic. | ONNX surgery regression test; prompt and track parsing proceeded. |
| `b378e6c` | Switched to bounded `dynamic_shapes` and export samples larger than one for object graphs; made the camera encoder statically batch 1. | Point and box ONNX inputs and outputs now carry real batch symbols. |
| `b6aa514` | Benchmarked on a non-default CUDA stream. | Removes TensorRT's default-stream synchronization warning and overhead. |
| `c24314b` | Limited track profiles to 1/2/4 and split 5-8 object groups into at most two launches. | 17 Python tests and the C++ state-selection test pass. |

## PACE experiment history

All GPU jobs below used QOS `embers`; no `inferno` resources were used.

| Job | GPU | Result | Finding or action |
| --- | --- | --- | --- |
| `11291365` | L40S | Failed | Initial ONNX export exposed unsupported complex-valued RoPE. Replaced it with the real-valued equivalent. |
| `11313493` | H100 | Failed | Four ONNX files exported; encoder build exposed a missing dynamic optimization profile. |
| `11313495` | RTX PRO Blackwell | Failed | PyTorch CUDA 12.8 environment could not initialize cuBLAS on this node. This is an environment/GPU compatibility issue, not a SAM2 graph result. |
| `11313554` | RTX6000 | Failed after export | Four ONNX files exported; TensorRT 11 had removed `EXPLICIT_BATCH`. |
| `11313682` | RTX6000 | Failed | TensorRT required an optimization profile for the dynamic encoder. |
| `11314456` | H100 | Failed | Encoder built; sizes-based Resize incorrectly mixed dynamic batch into linear resize dimensions. |
| `11314453` | RTX6000 | Partial | All four engines built. Point batches 1/2/4/8 worked, but box and track were internally fixed to batch 1 by a batch-1 export trace. |
| `11314620` | RTX6000 | Completed diagnostic | Confirmed four nominal profiles but fixed batch-1 box/track graph dimensions; measured track batch 1. |
| `11314760` | H100 | Partial | Bounded dynamic export fixed point/box. Encoder, point, and box built; track failed only on worst-case batch-8 profile after profiles 1/2/4 found tactics. |
| `11312337` | L40S | Pending | Full TensorRT build and same-GPU latency run. |
| `11313494` | A100 | Pending | Full TensorRT build and latency run. |
| `11314743` | RTX6000 | Pending | Full V0.3 export/build/latency run. |
| `11314933` | H100 | Completed | Same-checkpoint PyTorch H100 image-predictor baseline: 19.0847 ms predictor and 20.1773 ms total pipeline. |
| `11315064` | H100 | Completed | Track profiles 1/2/4 built and every requested engine microbenchmark completed. |
| `11315246` | H100 | Measurement completed | Exact matching PyTorch export-graph TF32 result was fully written. The scheduler records cancellation because duplicate cleanup raced with the completed epilog. |

## Partial TensorRT measurements

Job `11314453` used a Quadro RTX6000. That GPU does not support TF32, and
TensorRT explicitly disabled TF32, so these numbers are FP32 functional smoke
results rather than a final performance comparison. They were also captured
before the non-default-stream benchmark correction.

| Graph | Batch | Mean latency | Object throughput |
| --- | ---: | ---: | ---: |
| Encoder | 1 | 37.7332 ms | 26.50/s |
| Point prompt | 1 | 3.1575 ms | 316.71/s |
| Point prompt | 2 | 5.6940 ms | 351.24/s |
| Point prompt | 4 | 9.9161 ms | 403.38/s |
| Point prompt | 8 | 18.6988 ms | 427.84/s |
| Box prompt | 1 | 3.0397 ms | 328.98/s |
| Track | 1 | 44.3594 ms | 22.54/s |

No acceleration percentage is claimed from this table because the PyTorch
baseline is from an L40S and the TensorRT result is from an RTX6000.

Job `11315064` produced the first complete H100 TensorRT TF32 engine timings.
These measure preallocated graph execution on a non-default CUDA stream:

| Graph | Batch | Mean latency | Object throughput |
| --- | ---: | ---: | ---: |
| Encoder | 1 | 4.9073 ms | 203.78/s |
| Point prompt | 1 | 0.9793 ms | 1021.09/s |
| Point prompt | 2 | 1.4336 ms | 1395.04/s |
| Point prompt | 4 | 2.2793 ms | 1754.95/s |
| Point prompt | 8 | 4.0994 ms | 1951.51/s |
| Box prompt | 1 | 0.9824 ms | 1017.94/s |
| Box prompt | 2 | 1.4138 ms | 1414.59/s |
| Box prompt | 4 | 2.2429 ms | 1783.38/s |
| Box prompt | 8 | 4.0256 ms | 1987.31/s |
| Track | 1 | 4.5667 ms | 218.98/s |
| Track | 2 | 8.1348 ms | 245.86/s |
| Track | 4 | 14.9030 ms | 268.40/s |

Encoder plus initial point graph execution is 5.8866 ms, or 169.88 graph
executions per second. Comparing that sum directly with the standard H100
image-predictor result suggests a 69.2% latency reduction (3.24x), but this is
explicitly provisional: the predictor includes preprocessing/postprocessing and
uses a different mask-selection policy.

Job `11315246` supplied the required matching PyTorch denominator. It used the
same H100, FP32 graph modules, batch 1, real-valued RoPE, TF32 setting, synthetic
input shapes, 20 warmups, 100 runs, CUDA events, and a non-default CUDA stream.

| Graph | PyTorch | TensorRT | Latency reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 7.0237 ms | 4.9073 ms | 30.13% | 1.43x |
| Point prompt | 4.4223 ms | 0.9793 ms | 77.85% | 4.52x |
| Box prompt | 4.5581 ms | 0.9824 ms | 78.45% | 4.64x |
| Track | 17.0547 ms | 4.5667 ms | 73.22% | 3.73x |
| Encoder + point prompt | 11.4460 ms | 5.8866 ms | 48.57% | 1.94x |
| Encoder + track | 24.0784 ms | 9.4739 ms | 60.65% | 2.54x |

These are valid same-graph execution speedups. They are not yet end-to-end
camera-pipeline speedups and do not establish accuracy parity.

## Current interpretation and next experiments

The export/build path is now demonstrated for every graph, and genuine dynamic
batching is demonstrated for prompt graphs. The remaining immediate check is a
usable track engine with profiles 1/2/4. Splitting a batch of eight objects into
two batch-4 launches changes scheduling only, not model arithmetic or masks.

With the first matching performance check complete:

1. Add a real-input TensorRT-vs-PyTorch output-parity runner; synthetic engine
   timing alone is not an accuracy result.
2. Evaluate FP32 without TF32 first, then TF32 and FP16. Promote only the fastest
   candidate that passes the mask-level and dataset-level accuracy gates.
3. Repeat the winning configuration on L40S for a same-GPU comparison and then
   rebuild all plans on Thor.
4. On Thor, measure `camera -> preprocess -> encoder -> tail -> mask -> publish`
   with queue depth 1, pinned input buffers, one non-blocking CUDA stream, and no
   avoidable host/device copies.
