# Native runtime and ROS camera

[Guide index](../thor_testing_guide.md) · [Runtime architecture](../architecture.md)

## 8. 建置 C++ runtime 與 ROS workspace

先 build/install core library：

```bash
cd "$SAM2_TRT_ROOT"

cmake -S cpp -B build/core -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_INSTALL_PREFIX="$SAM2_TRT_ROOT/build/install"

cmake --build build/core -j"$(nproc)"
ctest --test-dir build/core --output-on-failure
cmake --install build/core

export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果 CMake 找不到 TensorRT，先確認 `NvInfer.h` 和 `libnvinfer.so` 是同一套
JetPack packages。只有在 JetPack 使用非標準位置時才顯式傳：

```bash
cmake -S cpp -B build/core -G Ninja \
  -DTENSORRT_INCLUDE_DIR=/actual/path/to/include \
  -DTENSORRT_LIBRARY=/actual/path/to/libnvinfer.so \
  -DCMAKE_INSTALL_PREFIX="$SAM2_TRT_ROOT/build/install"
```

再 build ROS packages：

```bash
cd "$SAM2_TRT_ROOT/ros_ws"
colcon build --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SAM2_TRT_ROOT/build/install"

source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
ros2 pkg executables sam2_trt_ros
ros2 interface show sam2_trt_msgs/srv/AddObject
```

預期 executable 是：

```text
sam2_trt_ros sam2_trt_node
```

安裝的 launch files 是：

- `camera_stream.launch.py`：只啟動 TensorRT node，接已存在的任意
  `sensor_msgs/Image` topic，適合 video publisher 或另外啟動的 camera driver；
- `realsense.launch.py`：同時啟動 `realsense2_camera` color stream 與 TensorRT node。

新 terminal 除了 source benchmark helper，還要加入 core library 與本 ROS
workspace：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
cd "$SAM2_TRT_ROOT"
```

## 9. 先用 recorded video 做 ROS smoke

此 TensorRT repo 沒有自己的 video publisher；沿用
`EfficientSAM3-Benchmark` 的 `video_stream_node`。

Terminal A：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:="$BENCH_ROOT/videos/test1.mov" \
  -p image_topic:=/image \
  -p fps:=0.0 \
  -p playback_rate:=1.0 \
  -p frame_id:=video \
  -p resize_width:=640
```

Terminal B：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"

mkdir -p "$SAM2_TRT_ROOT/results/thor/video_smoke"
ros2 launch sam2_trt_ros camera_stream.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-hiera-tiny/fp32" \
  precision:=fp32 \
  image_topic:=/image \
  trace_path:="$SAM2_TRT_ROOT/results/thor/video_smoke/runtime.jsonl"
```

`camera_stream.launch.py` 不會啟動 camera driver；`image_topic` 可以是任意
`sensor_msgs/Image` topic。Node 接受 `rgb8` 或 `bgr8`。

Terminal C：先確認 node 正常收到 frames，再新增一個 point：

```bash
ros2 topic hz /sam/result_json
ros2 topic echo /sam/result_json --once

ros2 service call /sam/add_object sam2_trt_msgs/srv/AddObject \
  "{kind: 0, x0: 320.0, y0: 240.0, x1: 0.0, y1: 0.0}"

ros2 topic hz /segmentation_mask
ros2 topic echo /sam/result_json --once
```

新增 box 或 reset：

```bash
ros2 service call /sam/add_object sam2_trt_msgs/srv/AddObject \
  "{kind: 1, x0: 180.0, y0: 120.0, x1: 460.0, y1: 390.0}"

ros2 service call /sam/reset std_srvs/srv/Trigger '{}'
```

Prompt coordinates 是原始 camera/video frame pixels，runtime 會縮放到 1024 model
input。輸入必須是 `rgb8` 或 `bgr8`。

可用 `rqt_image_view` 檢查第一個 object mask：

```bash
ros2 run rqt_image_view rqt_image_view /segmentation_mask
```

若未安裝 `rqt_image_view`，也可用 `ros2 bag record` 保存結果後離線檢查：

```bash
mkdir -p "$SAM2_TRT_ROOT/results/thor/ros_smoke"
ros2 bag record \
  /image \
  /segmentation_mask \
  /sam/object_masks \
  /sam/result_json \
  -o "$SAM2_TRT_ROOT/results/thor/ros_smoke/test1"
```

## 10. RealSense camera 測試

一般 smoke 可在同一個 terminal 一鍵啟動 camera 與 TensorRT node：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
mkdir -p "$SAM2_TRT_ROOT/results/thor/realsense"

ros2 launch sam2_trt_ros realsense.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-hiera-tiny/fp32" \
  precision:=fp32 \
  trace_path:="$SAM2_TRT_ROOT/results/thor/realsense/runtime.jsonl"
```

這個 launch 開啟 color、關閉 depth，使用 driver 自己選定的 color profile。若要
指定或診斷 camera profile，則依下列兩-terminal 流程分開啟動。

Terminal A 啟動 color stream：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false
```

不要假設 camera 支援某個 profile。先查 driver 實際接受值：

```bash
ros2 param describe /camera/camera rgb_camera.color_profile
ros2 param get /camera/camera rgb_camera.color_profile
rs-enumerate-devices -s
lsusb -t
```

`lsusb -t` 中 `480M` 是 USB 2；`5000M` 或更高才是 USB 3。若 driver 將要求的
profile fallback，例如從 30 FPS 降成 15 FPS，benchmark 必須記錄最後實際值。

確認實際 resolution 與 rate：

```bash
ros2 topic echo --once /camera/camera/color/camera_info | grep -E 'width:|height:'
ros2 topic hz /camera/camera/color/image_raw
```

Terminal B 使用同一 bundle：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"

mkdir -p "$SAM2_TRT_ROOT/results/thor/realsense"
ros2 launch sam2_trt_ros camera_stream.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-hiera-tiny/fp32" \
  precision:=fp32 \
  image_topic:=/camera/camera/color/image_raw \
  trace_path:="$SAM2_TRT_ROOT/results/thor/realsense/runtime.jsonl"
```

Terminal C 加 prompt、看 mask rate 與掉幀：

```bash
ros2 service call /sam/add_object sam2_trt_msgs/srv/AddObject \
  "{kind: 0, x0: 320.0, y0: 240.0, x1: 0.0, y1: 0.0}"

ros2 topic hz /sam/object_masks
ros2 topic hz /sam/result_json
ros2 topic echo /sam/result_json
```

`/sam/result_json` 範例：

```json
{"stamp_ns":123456789,"frame_index":42,"objects":[1,2],"queue_wait_ms":0.031,"color_convert_ms":0.481,"inference_ms":21.732,"mask_publish_ms":0.109,"worker_total_ms":22.322,"callback_total_ms":22.353,"frame_interval_ms":33.333,"processing_capacity_fps":44.799,"processed_fps":30.000,"tracking_fps":30.000,"dropped":0,"dropped_frames":17,"source_age_ms":24.126,"end_to_end_ms":24.126}
```

`dropped_frames` 是 latest-frame slot 被新 frame 覆寫的累計數。Camera FPS 高於
inference FPS 時掉幀是預期行為；`dropped` 是自上一個已處理 frame 後新增的掉幀數。
Queue depth 1 的目的是避免追蹤舊畫面。

## 11. ROS interfaces 與多物件行為

Topics：

| Name | Type | 說明 |
| --- | --- | --- |
| `/segmentation_mask` | `sensor_msgs/Image` mono8 | 第一個 object 的 compatibility mask |
| `/sam/object_masks` | `sensor_msgs/Image` mono8 | 每個 object 一則；ID 附在 `header.frame_id` |
| `/sam/result_json` | `std_msgs/String` | input stamp、object IDs、runtime timings、FPS 與 dropped frames |

Services：

| Name | Type | 說明 |
| --- | --- | --- |
| `/sam/add_object` | `sam2_trt_msgs/srv/AddObject` | `kind=0` point，`kind=1` box |
| `/sam/reset` | `std_srvs/srv/Trigger` | 清除所有 objects 與 memory state |

最多八個 objects。Prompt graph profiles 是 1/2/4/8；track profiles 是 1/2/4。
當同一 memory-length bucket 有 5–8 objects，runtime 自動切成兩組執行，不需使用者
介入。

`sam2_trt_interactive_viewer` 提供與既有 ROS benchmark 相同的 OpenCV 操作：

- 左鍵點一下：新增 point-prompt object；
- 按住左鍵拖曳再放開：新增 box-prompt object；
- `r`：呼叫 `/sam/reset` 清除所有 objects；
- `q` 或 `Esc`：關閉 viewer。

Viewer 只把 prompt 送到既有 services，並訂閱 camera、mask 與 result topics。
TensorRT 推論仍完全在 C++ node；畫面上會顯示 result output FPS、
`inference_ms`、`worker_total_ms`、`source_age_ms` 與 dropped frames。

在 Thor 本機桌面用一個 command 啟動 RealSense、tracker 與互動 viewer：

```bash
cd "$SAM2_TRT_ROOT"
source "$HOME/EfficientSAM3-Benchmark/scripts/source_thor_ros_env.sh"
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$SAM2_TRT_ROOT/results/thor/tv5_interactive_001"
ros2 launch sam2_trt_ros interactive_realsense.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-tinyvit-5m/fp16_aux0" \
  precision:=fp16 \
  trace_path:="$SAM2_TRT_ROOT/results/thor/tv5_interactive_001/runtime.jsonl"
```

要量模型的實際 capacity，使用 D455F 支援的最高解析度 60 FPS profile：

```bash
ros2 launch sam2_trt_ros interactive_realsense.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-tinyvit-5m/fp16_aux0" \
  precision:=fp16 \
  color_profile:=848x480x60 \
  track_concurrency:=8 \
  pipeline_overlap:=true \
  replace_on_prompt:=false \
  trace_path:="$SAM2_TRT_ROOT/results/thor/tv5_60fps/runtime.jsonl"
```

不要使用 `1280x720x60` 或 `960x540x60`；這台 D455F 不支援，driver 會退回
`1280x720x30`。啟動後應在 log 看到：

```text
Open profile: stream_type: Color(0), Format: RGB8, Width: 848, Height: 480, FPS: 60
Device USB type: 3.2
```

`pipeline_overlap:=true` 會在追蹤 frame N 時同時 encode frame N+1，最大化
throughput。JSONL 會記錄 `"pipeline_overlap":true` 與
`"pipeline_delay_frames":1`。它不改變 mask 計算，但固定增加一個 processed
frame 的 source-age；需要最低即時延遲時使用 `pipeline_overlap:=false`。

從 SSH 啟動到 Thor 的既有桌面時，另外設定該桌面的 `DISPLAY`、
`XDG_RUNTIME_DIR` 與 `DBUS_SESSION_BUS_ADDRESS`。`q` 只關閉 viewer；
在 launch terminal 按 `Ctrl+C` 才會一起停止 camera 與 tracker。預設每個新
click/drag 會先 reset 再建立單一 object，符合互動選取時反覆改 prompt 的
預期。需要最多八個 objects 的累加模式時傳
`replace_on_prompt:=false`。

Viewer 依 `header.stamp` 配對 image、result 與所有 object masks；同一 frame
的預期 masks 全部到齊後才提交新 overlay。不要在 result 先到時先顯示 raw
frame，否則 raw/mask 會交替造成視覺閃爍。

畫面上的 `infer` 是完整 `Tracker::process_rgb8` wall time：包含 host-to-device
copy、CUDA resize/normalize、encoder、prompt/track engine、memory state packing、
mask resize 與 device-to-host copy，但不含 ROS queue wait 或 mask publish。
它是 live pipeline 最實用的 model-path latency，不等於單張 TensorRT plan 的
kernel latency。純 engine latency 請用第 6 節的 `benchmark-engine`；其 inputs
和 outputs 預先配置，使用 CUDA events，只量 plan execution。

SAM2 steady tracking 每幀由一個 encoder 加上每個 object 的 track step 組成。
目前 Thor scheduler 使用 batch-1 track step，因此可用
`encoder mean + object_count * track-b1 mean` 估計純 engine 下限，再用
`inference_ms` 量 allocation、state packing、transfer 與 postprocess 加入後的
實際 model path。
