# Thor deployment and acceptance runbook

完整的 environment、Thor paths、engine commands、recorded-video/RealSense ROS
smoke 與 troubleshooting 請先看
[thor_testing_guide.md](thor_testing_guide.md)。本頁保留精簡的 acceptance reference。

## 1. Inputs and provenance

Record these before exporting:

- model ID and exact checkpoint path;
- SHA256 for the encoder and downstream checkpoint;
- official SAM2 and distillation repository commits;
- Thor probe and pinned environment lock;
- SA-V and image-evaluation dataset revisions.

The exporter writes checkpoint/source/environment information to `manifest.json`.
`sam2-trt verify-bundle` re-hashes the checkpoint and every plan before launch.

## 2. Build the native runtime on Thor

Install the core to a local prefix so the ROS workspace can find it:

```bash
cmake -S cpp -B build/core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install"
cmake --build build/core -j"$(nproc)"
ctest --test-dir build/core --output-on-failure
cmake --install build/core

source /opt/ros/jazzy/setup.bash
cd ros_ws
colcon build --symlink-install \
  --cmake-args -DCMAKE_PREFIX_PATH="$OLDPWD/build/install"
source install/setup.bash
```

If the probe reports a ROS distribution other than Jazzy, use that distribution's setup
file and record it in the lock. Do not silently mix ROS distributions.

## 3. Accuracy evaluation

Run the existing PyTorch benchmark repository with FP32 and the exact same checkpoint to
produce:

- full SA-V J&F in percentage points;
- image mIoU in percentage points;
- an NPZ containing one binary array per `frame/object` key.

Run the identical prompts, frames, object-add times, and threshold with the TensorRT
candidate. Each report has this schema:

```json
{
  "metric_unit": "percentage_points",
  "metrics": {"sav_jf": 80.0, "image_miou": 75.0},
  "binary_masks_npz": "binary_masks.npz"
}
```

Apply and store the gate in the bundle:

```bash
sam2-trt validate \
  --baseline results/pytorch_fp32/report.json \
  --candidate results/tensorrt/report.json \
  --bundle-dir bundles/MODEL/PRECISION \
  --output results/tensorrt/accuracy_gate.json
```

Exit code 0 means all three conditions pass. Exit code 2 means the candidate is rejected.
The 0.1 limit is interpreted as a percentage-point drop, not a relative percentage.

## 4. ROS 2 RealSense integration

For the default color stream, launch the RealSense driver and TensorRT node together:

```bash
ros2 launch sam2_trt_ros realsense.launch.py \
  bundle_dir:=/absolute/path/to/bundle \
  precision:=fp32 \
  trace_path:=/absolute/path/to/results/runtime.jsonl
```

If the camera driver or a video publisher is already running, use the generic launch so
it is not started twice:

```bash
ros2 launch sam2_trt_ros camera_stream.launch.py \
  bundle_dir:=/absolute/path/to/bundle \
  precision:=fp32 \
  image_topic:=/camera/camera/color/image_raw \
  trace_path:=/absolute/path/to/results/runtime.jsonl
```

Add a positive point in camera-pixel coordinates:

```bash
ros2 service call /sam/add_object sam2_trt_msgs/srv/AddObject \
  "{kind: 0, x0: 640.0, y0: 360.0, x1: 0.0, y1: 0.0}"
```

Add a box or reset all state:

```bash
ros2 service call /sam/add_object sam2_trt_msgs/srv/AddObject \
  "{kind: 1, x0: 300.0, y0: 200.0, x1: 800.0, y1: 650.0}"
ros2 service call /sam/reset std_srvs/srv/Trigger '{}'
```

The standalone node can switch between compatible bundles without restarting
ROS. It builds the replacement tracker first and swaps it in only after every
TensorRT engine loads successfully; switching clears the tracking state.

```bash
ros2 service call /sam/switch_model sam2_trt_msgs/srv/SwitchModel \
  "{model_id: sam2.1-tinyvit-11m, bundle_dir: '$SAM2_TRT_ROOT/bundles/sam2.1-tinyvit-11m/fp16_aux0', precision: fp16}"
```

The response records `load_ms` and the active model. Keeping only one tracker
resident avoids duplicating the large prompt/track TensorRT contexts shared by
the 5M, 11M, and 21M variants.

Topics:

- `/segmentation_mask`: first object's `mono8` mask for compatibility;
- `/sam/object_masks`: one `mono8` message per object, with object ID appended to
  `header.frame_id`;
- `/sam/result_json`: stamp, object IDs, node-level timing, tracking FPS, and frames
  overwritten by latest-frame scheduling.

The runtime intentionally keeps corrective clicks, mask prompts, visualization, and
NITROS out of the C++ inference critical path. Adding one point or box creates one
new object.

For live click/drag prompts and a mask/timing overlay on the Thor desktop, source
the shared ROS venv and use the integrated launch:

```bash
source "$HOME/EfficientSAM3-Benchmark/scripts/source_thor_ros_env.sh"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
ros2 launch sam2_trt_ros interactive_realsense.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-tinyvit-5m/fp16_aux0" \
  precision:=fp16 \
  trace_path:="$SAM2_TRT_ROOT/results/thor/tv5_interactive_001/runtime.jsonl"
```

Click for a point, drag for a box, press `r` to reset all objects, and press
`q`/`Esc` to close the viewer. Visualization is a separate ROS subscriber;
the C++ TensorRT trace remains the latency source of truth. A new prompt
replaces the current object by default; use `replace_on_prompt:=false` only
for intentional multi-object accumulation. The viewer waits for every mask
listed by the same-stamp result before swapping the displayed overlay.
The overlay separates three rates:

- `tracker`: result rate computed from source frame stamps;
- `UI-rx`: result callback arrival rate in the Python viewer;
- `present`: rate at which a new completed overlay is actually shown.

`compose` is mask blending time and `display` is the OpenCV present path. These
numbers make a display bottleneck visible without confusing it with
`inference_ms`. Contour drawing is disabled by default because it does not
change the model mask and adds CPU work; enable it only when needed with
`draw_contours:=true`.

## 5. Performance acceptance

Set `trace_path` in either launch to capture node-level queue, color conversion, combined
TensorRT inference, mask publication, source-to-result, frame interval, FPS, and
dropped-frame timings. Summarize with:

```bash
sam2-trt benchmark --trace results/runtime.jsonl --output results/runtime_summary.json
```

`processing_capacity_fps` is the reciprocal of per-frame `worker_total_ms`
(processing after dequeue, excluding `queue_wait_ms`), while
`processed_fps` is the observed interval between processed frames. They are expected to
differ when camera/USB delivery is slower or irregular. Use summary `throughput_fps`
(`interval_count / measurement_duration_s`) for the measured run FPS; do not average
instantaneous FPS values.

Test 1/2/4/8 active objects, point and box initialization, and objects introduced after
tracking has started. Record power mode, clocks, camera resolution/FPS, thermal state,
precision, model/checkpoint hash, and bundle manifest. Select double-stream overlap or
CUDA Graph capture only after a measured Thor comparison; early dynamic-memory frames
must continue through normal `enqueueV3`.

The trace does not yet split the combined `inference_ms` into encoder, tail, and
postprocess. `end_to_end_ms` is emitted only for a nonzero input stamp in the same ROS
clock domain as the node.

## Known validation boundary

The repository contains the complete build/runtime path, but plans and camera results are
device-specific. A release is not accepted until the target Thor has successfully built
all four plans, passed full accuracy gates, built the ROS workspace, and completed the
camera matrix above. The fused resize kernel uses bilinear sampling; compare it against
the PyTorch oracle for the selected RealSense resolution. If torchvision antialiasing for
a downsampled camera feed breaches the gate, replace that stage with an accuracy-matched
separable antialias kernel before considering lower precision.
