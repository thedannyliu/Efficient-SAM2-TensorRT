# SAM2 TensorRT: stateful video segmentation on Jetson Thor

A Python export toolchain and C++/CUDA runtime for deploying SAM2.1 and distilled TinyViT encoders through ROS 2. The engineering focus is preserving per-object tracking state while reducing inference latency on embedded hardware.

**Start here:** [measured results](#measured-results) · [runtime design](#runtime-design) · [CPU checks](#run-the-cpu-checks) · [Thor deployment guide](docs/thor_testing_guide.md)

## Measured results

The committed [Thor experiment record](docs/thor_optimization_experiments.md#same-checkpoint-pytorch-versus-tensorrt-camera-ab) reports the following July 24, 2026 comparison: Jetson AGX Thor, 120 W, TensorRT 10.13.3.9, TinyViT-5M, one object, center-point prompt, RealSense 1280×720 RGB8 at 30 FPS, and 100 steady tracking frames per runtime.

| Runtime | Camera-pipeline throughput | Mean model/pipeline latency | Mean source-to-result age |
| --- | ---: | ---: | ---: |
| PyTorch FP32 | 3.840 FPS | 177.542 ms | 312.423 ms |
| TensorRT FP16 | 26.505 FPS | 34.080 ms | 80.990 ms |

This recorded run shows **6.90× throughput** and **5.21× lower measured latency**. The timing scopes differ: PyTorch measures the online model step; TensorRT also includes transfers, preprocessing, state/mask conversion, and synchronization. Raw traces are retained on the device and are not included in this repository. These are historical measurements, not results reproduced by the CPU checks below.

Accuracy is evaluated separately. The [July 29 checkpoint refresh](docs/thor_optimization_experiments.md#2026-07-29-tinyvit-best-checkpoint-refresh) reports TinyViT-5M mean mask agreement of **0.99595 for point prompts** and **0.99192 for box prompts**, using eight frame/prompt samples per mode against the matching PyTorch checkpoint. That refresh uses different checkpoint hashes from the July 24 speed comparison; the two records do not establish a joint speed/accuracy result for one bundle. Model agreement is not ground-truth segmentation accuracy or full-video tracking validation.

## What this repository adds

SAM2's pretrained model and architecture come from [Meta's SAM2](https://github.com/facebookresearch/sam2). TinyViT training comes from the companion [SAM2 distillation pipeline](https://github.com/thedannyliu/SAM2-Distillation-Pipeline). This repository supplies the deployment layer:

- Four exported graphs: image encoder, point initialization, box initialization, and temporal tracking.
- A C++/CUDA runtime that packs memory and object pointers, executes TensorRT engines, and maintains independent object state.
- A ROS 2 latest-frame worker, add/reset controls, and an interactive viewer that matches masks to the correct image timestamp.
- Checkpoint and engine hashes, accuracy-report validation, and per-frame timing traces.
- Measured scheduler experiments covering batch size, concurrent execution contexts, and cross-frame overlap.

## Runtime design

The image encoder runs once per frame. An initial point or box creates object state; subsequent tracking selects temporal memories and pointers independently for each object. Results feed both the mask publisher and the next tracking step.

| Engineering decision | Reason and evidence |
| --- | --- |
| Separate prompt and tracking graphs | Initialization and temporal propagation have different input/state contracts. |
| Batch-1 tracking by default | Thor measurements found batch-2/4 tracking slower per object; larger batches remain optional. |
| Latest-frame camera processing | Avoids an accumulating queue when the camera produces frames faster than the tracker can process them. |
| Record source age as well as FPS | Cross-frame overlap can improve throughput while making displayed results older. |
| Match displayed masks by timestamp | Prevents the viewer from alternating an unmasked frame with its delayed mask. |

For examples of rejected optimizations and their measurements, see the [experiment record](docs/thor_optimization_experiments.md).

## Run the CPU checks

These tests exercise state selection, export helpers, ONNX rewrites, manifest handling, and validation logic. They do not require model weights, TensorRT, ROS, or a GPU.

```bash
git clone https://github.com/thedannyliu/Efficient-SAM2-TensorRT.git
cd Efficient-SAM2-TensorRT
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
g++ -std=c++20 cpp/tests/state_selection_test.cpp -Icpp/include -o /tmp/sam2_state_test
/tmp/sam2_state_test
```

## Deploy on Thor

Start with the [complete deployment guide](docs/thor_testing_guide.md), which covers dependencies, checkpoint locations, engine builds, camera input, and measurement. The [short runbook](docs/thor_runbook.md) covers acceptance checks.

```bash
sam2-trt probe --output results/thor_probe.json
sam2-trt pin --probe results/thor_probe.json --output environment.lock.json
```

Build TensorRT plans on the target GPU from the intended checkpoint. Plans and model weights are not bundled. PACE supports development and oracle evaluation; its engines and timings are not substitutes for Thor deployment measurements.

### Accuracy gates and current limits

The strict validator in [`sam2_trt/validate.py`](sam2_trt/validate.py) defaults to at most 0.1 percentage-point degradation in SA-V J&F and image mIoU, plus at least 0.999 IoU for every saved frame/object mask. The Thor optimization log separately uses a mean model-agreement threshold of 0.95 for its sampled prompt experiments. Passing that local experiment threshold does **not** establish passage of the strict dataset-level gate.

The July 29 TinyViT-21M point experiment includes one multimask-selection outlier (minimum IoU 0.79071). Multi-object throughput, camera delivery, and viewer overhead vary with the configuration. Consult the exact run, checkpoint, prompt type, and timing scope before reusing a result.

## Code map

| Path | Purpose |
| --- | --- |
| [`sam2_trt/export.py`](sam2_trt/export.py), [`build.py`](sam2_trt/build.py) | ONNX export and TensorRT build configuration |
| [`sam2_trt/state.py`](sam2_trt/state.py) | Reference memory and pointer selection |
| [`cpp/src/tracker.cpp`](cpp/src/tracker.cpp) | Stateful C++ tracking and execution scheduling |
| [`ros_ws/src/sam2_trt_ros`](ros_ws/src/sam2_trt_ros) | ROS node, launch files, and viewer |
| [`tests`](tests), [`cpp/tests`](cpp/tests) | CPU logic and state-selection checks |
| [`docs/pace_experiments.md`](docs/pace_experiments.md) | Export investigations and server-side measurements |

## Related work in this portfolio

- [EfficientSAM3-Benchmark](https://github.com/thedannyliu/EfficientSAM3-Benchmark): shared dataset protocols and PyTorch benchmark backends.
- [SAM2-Distillation-Pipeline](https://github.com/thedannyliu/SAM2-Distillation-Pipeline): encoder training and integration.
- [Efficient-SAM3-TensorRT](https://github.com/thedannyliu/Efficient-SAM3-TensorRT): SAM3 deployment experiments and the unified viewer.

Upstream model code, checkpoints, and dependencies retain their own license terms.
