# Thor deployment and acceptance runbook

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

Start the RealSense color stream and then launch:

```bash
ros2 launch sam2_trt_ros realsense.launch.py \
  bundle_dir:=/absolute/path/to/bundle \
  precision:=fp32 \
  image_topic:=/camera/camera/color/image_raw
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

Topics:

- `/segmentation_mask`: first object's `mono8` mask for compatibility;
- `/sam/object_masks`: one `mono8` message per object, with object ID appended to
  `header.frame_id`;
- `/sam/result_json`: stamp, object IDs, and count of frames overwritten by latest-frame
  scheduling.

The first version intentionally omits corrective clicks, mask prompts, display/overlay on
the critical path, and NITROS. Adding one point or box creates one new object.

## 5. Performance acceptance

Capture JSONL rows with capture, preprocess, encoder, tail, postprocess, end-to-end, frame
interval, and dropped-frame timings. Summarize with:

```bash
sam2-trt benchmark --trace results/runtime.jsonl --output results/runtime_summary.json
```

Test 1/2/4/8 active objects, point and box initialization, and objects introduced after
tracking has started. Record power mode, clocks, camera resolution/FPS, thermal state,
precision, model/checkpoint hash, and bundle manifest. Select double-stream overlap or
CUDA Graph capture only after a measured Thor comparison; early dynamic-memory frames
must continue through normal `enqueueV3`.

## Known validation boundary

The repository contains the complete build/runtime path, but plans and camera results are
device-specific. A release is not accepted until the target Thor has successfully built
all four plans, passed full accuracy gates, built the ROS workspace, and completed the
camera matrix above. The fused resize kernel uses bilinear sampling; compare it against
the PyTorch oracle for the selected RealSense resolution. If torchvision antialiasing for
a downsampled camera feed breaches the gate, replace that stage with an accuracy-matched
separable antialias kernel before considering lower precision.
