# Jetson Thor 測試與使用指南

本文件說明如何在 Jetson Thor 上建置、測試及使用本 repo 的 SAM2
TensorRT runtime。Thor 的 ROS、Python environment、camera topic 與 checkpoint
layout 沿用 `EfficientSAM3-Benchmark` 的
`docs/thor_ros_camera_benchmark.md`；不要在 Thor 上使用 PACE 的
`/storage/home/...` 或 `/storage/project/...` 路徑。

## 0. 目前可以驗證什麼

目前 Thor 路徑包含：

- SAM2.1 Hiera Tiny/Small/Base+/Large 與 distilled TinyViT 21M/11M/5M ONNX export；
- 在目標 Thor 上建立 FP32、TF32、FP16 或 BF16 TensorRT plans；
- TensorRT graph microbenchmark；
- C++ CUDA preprocessing、encoder、point/box prompt、memory tracking 與 mask postprocess；
- ROS 2 Jazzy latest-frame camera subscriber、通用 image-topic launch 與一鍵 RealSense launch；
- 最多八個 objects，point/box service、per-object masks、reset、dropped-frame count
  與逐幀 JSONL runtime trace。

目前尚未接完的部分也必須先知道：

- ROS node 尚未發布 `/sam/overlay` 或 `/segmented_image`；`enable_overlay`
  parameter 目前只是保留介面。
- `/sam/result_json` 與可選的 JSONL trace 已包含 queue、RGB/BGR conversion、整體
  TensorRT inference、mask publish、pipeline、source-to-result latency、tracking FPS
  與掉幀；尚未把整體 inference 再拆成 encoder/tail/postprocess。
- `source_age_ms`/`end_to_end_ms` 只有在 camera message 提供非零且與 node 使用
  同一 ROS clock 的 timestamp 時才會出現。
- accuracy gate 工具已存在，但 TensorRT real-input report/NPZ 產生器仍需接上。
  未產生並通過該報告前，不得宣稱「不掉精度」。

第一次 bring-up 建議嚴格依此順序：

```text
environment/import probe
  -> Hiera Tiny FP32 export + build + verify
  -> four-engine synthetic smoke
  -> C++/ROS build
  -> recorded-video ROS smoke
  -> RealSense one-object smoke
  -> multi-object point/box/reset smoke
  -> real-input accuracy gate
  -> TF32/FP16/BF16 candidates
```

## 1. Thor 目錄配置

建議沿用既有 Thor benchmark repo 與 venv，TensorRT repo 另外放一個小型 checkout：

```text
~/EfficientSAM3-Benchmark/                 # 既有 Thor benchmark/oracle repo
  scripts/source_thor_ros_env.sh
  external/sam2/
  external/SAM2-Distillation-Pipeline/
  checkpoints/sam2/
  checkpoints/distill/
  videos/test1.mov
  videos/test2.mov

~/Efficient-SAM2-TensorRT/                 # 本 repo
  bundles/                                 # ONNX + Thor-specific engines；不進 Git
  results/                                 # benchmark/accuracy outputs；不進 Git
  logs/                                    # local logs；不進 Git
  build/                                   # C++ build/install；不進 Git
  ros_ws/

~/venvs/effisam3_venv_ros/                 # 共用 Jetson/ROS Python environment
/opt/ros/jazzy/setup.bash
```

以下所有命令假設：

```bash
export BENCH_ROOT="$HOME/EfficientSAM3-Benchmark"
export SAM2_TRT_ROOT="$HOME/Efficient-SAM2-TensorRT"
export THOR_VENV="$HOME/venvs/effisam3_venv_ros"
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
export SAM3_SOURCE="$HOME/efficientsam3/sam3"
```

如果路徑不同，只改這些 variables。不要把 PACE 絕對路徑寫進 Thor bundle。

在 Thor clone 本 repo。不要複製 PACE 建立的 `.engine`；TensorRT plans 必須在
目標 Thor 上重建：

```bash
git clone git@github.com:thedannyliu/Efficient-SAM2-TensorRT.git \
  "$HOME/Efficient-SAM2-TensorRT"
```

## 2. 一次性系統與 Python environment setup

先依 NVIDIA 對目前 JetPack release 的文件安裝 JetPack 與 Jetson-compatible
PyTorch/torchvision。不要用 generic PyPI PyTorch 或 Ubuntu
`nvidia-cuda-toolkit` 取代 JetPack components。

確認平台：

```bash
cat /etc/os-release
uname -m
nvidia-smi
nvcc --version
python3 --version
```

預期 architecture 是 `aarch64`，device model 包含 Thor。實際 JetPack、CUDA、
TensorRT 與 PyTorch versions 以 probe 結果為準，不在文件中硬編版本號。

安裝 build 與 ROS packages：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build \
  python3-venv python3-opencv python3-colcon-common-extensions \
  ros-jazzy-ros-base \
  ros-jazzy-cv-bridge \
  ros-jazzy-realsense2-camera \
  ros-jazzy-realsense2-description \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  ros-jazzy-std-srvs \
  ros-jazzy-rosidl-default-generators \
  ros-jazzy-rqt-image-view
```

若 Thor 的 APT repository 沒有 `realsense2-camera` packages，依既有 benchmark
文件在獨立 ROS workspace build official `realsense-ros`，並在 source 本 repo 的
ROS workspace 前先 source 該 workspace。

JetPack 通常已提供 TensorRT runtime、headers 與 Python binding。確認它們
來自同一套 JetPack installation：

```bash
test -f /usr/include/aarch64-linux-gnu/NvInfer.h || \
  test -f /usr/local/tensorrt/include/NvInfer.h
ldconfig -p | grep nvinfer
python3 -c 'import tensorrt as trt; print(trt.__version__)'
```

若缺少 TensorRT，從已配置的 NVIDIA JetPack APT repository 補齊 TensorRT/
development packages；不要用不同版本的 x86/PyPI package 混裝。

依既有 benchmark 文件建立共用 venv；`--system-site-packages` 讓 venv 看得到
APT 安裝的 ROS、OpenCV 與 TensorRT：

```bash
python3 -m venv --system-site-packages "$THOR_VENV"

cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh

python -m pip install -U pip
python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml
python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil
python -m pip install onnx onnxscript
python -m pip install -e . --no-deps

cd "$SAM2_TRT_ROOT"
python -m pip install -e . --no-deps
```

不要在 Thor 直接安裝 PACE 的 `requirements.txt`。`numpy<2` 也要保留，因為
既有 Thor `cv_bridge` 是用 NumPy 1.x ABI 建立。

每個 Thor terminal 都用相同順序 source：

```bash
export BENCH_ROOT="$HOME/EfficientSAM3-Benchmark"
export SAM2_TRT_ROOT="$HOME/Efficient-SAM2-TensorRT"
export THOR_VENV="$HOME/venvs/effisam3_venv_ros"
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
export SAM3_SOURCE="$HOME/efficientsam3/sam3"

cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
cd "$SAM2_TRT_ROOT"
```

驗證 imports：

```bash
python - <<'PY'
import cv2
import rclpy
import cv_bridge
import onnx
import tensorrt as trt
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("TensorRT:", trt.__version__)
print("ONNX:", onnx.__version__)
PY
```

## 3. Probe 並固定 Thor environment

先記錄實際 environment：

```bash
cd "$SAM2_TRT_ROOT"
mkdir -p results/thor

sam2-trt probe --output results/thor/environment_probe.json
sam2-trt pin \
  --probe results/thor/environment_probe.json \
  --output environment.lock.json

cat results/thor/environment_probe.json
cat environment.lock.json
```

`pin` 會拒絕非 Thor device，也會要求 architecture、device model、TensorRT、
PyTorch CUDA 與 ROS distro 都存在。`environment.lock.json` 是 local artifact，
不應提交 Git。

另記錄：

```bash
git -C "$SAM2_TRT_ROOT" rev-parse HEAD
git -C "$BENCH_ROOT/external/sam2" rev-parse HEAD
git -C "$BENCH_ROOT/external/SAM2-Distillation-Pipeline" rev-parse HEAD
sha256sum "$BENCH_ROOT"/checkpoints/sam2/*.pt
sudo nvpmodel -q
```

若要固定最高 clocks，先記錄 power mode，再依該 Thor 的管理政策執行：

```bash
sudo jetson_clocks
jetson_clocks --show
```

不要假設某個 `nvpmodel -m` ID 在所有 Thor image 都相同。

## 4. Model sources 與 checkpoints

沿用 benchmark repo 的 sources：

```bash
test -d "$BENCH_ROOT/external/sam2"
test -d "$BENCH_ROOT/external/SAM2-Distillation-Pipeline"
```

若尚未準備：

```bash
cd "$BENCH_ROOT"
source scripts/source_thor_ros_env.sh
bash scripts/setup_model_repos.sh
bash scripts/download_sam2_family_checkpoints.sh

test -d external/SAM2-Distillation-Pipeline || \
  git clone git@github.com:thedannyliu/SAM2-Distillation-Pipeline.git \
    external/SAM2-Distillation-Pipeline
python -m pip install -e external/SAM2-Distillation-Pipeline --no-deps
```

`download_sam2_family_checkpoints.sh` 會下載 official Hiera checkpoints，不會下載
自行訓練的 TinyViT Stage1 checkpoints。21M/11M/5M `.pt` 必須從既有訓練 artifacts
複製到下表位置，並保存 SHA256。

本 repo registry 預期的模型與建議 Thor paths：

| Model ID | Encoder checkpoint | Downstream checkpoint |
| --- | --- | --- |
| `sam2.1-hiera-tiny` | `checkpoints/sam2/sam2.1_hiera_tiny.pt` | same |
| `sam2.1-hiera-small` | `checkpoints/sam2/sam2.1_hiera_small.pt` | same |
| `sam2.1-hiera-base-plus` | `checkpoints/sam2/sam2.1_hiera_base_plus.pt` | same |
| `sam2.1-hiera-large` | `checkpoints/sam2/sam2.1_hiera_large.pt` | same |
| `sam2.1-tinyvit-21m` | `checkpoints/distill/tv21.pt` | SAM2.1-L |
| `sam2.1-tinyvit-11m` | `checkpoints/distill/tv11.pt` | SAM2.1-L |
| `sam2.1-tinyvit-5m` | `checkpoints/distill/tv5.pt` | SAM2.1-L |

設定明確的 absolute paths：

```bash
export SAM2_HIERA_TINY_CHECKPOINT="$BENCH_ROOT/checkpoints/sam2/sam2.1_hiera_tiny.pt"
export SAM2_HIERA_SMALL_CHECKPOINT="$BENCH_ROOT/checkpoints/sam2/sam2.1_hiera_small.pt"
export SAM2_HIERA_BASE_PLUS_CHECKPOINT="$BENCH_ROOT/checkpoints/sam2/sam2.1_hiera_base_plus.pt"
export SAM2_HIERA_LARGE_CHECKPOINT="$BENCH_ROOT/checkpoints/sam2/sam2.1_hiera_large.pt"
export SAM2_TINYVIT_21M_CHECKPOINT="$BENCH_ROOT/checkpoints/distill/tv21.pt"
export SAM2_TINYVIT_11M_CHECKPOINT="$BENCH_ROOT/checkpoints/distill/tv11.pt"
export SAM2_TINYVIT_5M_CHECKPOINT="$BENCH_ROOT/checkpoints/distill/tv5.pt"

for checkpoint in \
  "$SAM2_HIERA_TINY_CHECKPOINT" \
  "$SAM2_HIERA_SMALL_CHECKPOINT" \
  "$SAM2_HIERA_BASE_PLUS_CHECKPOINT" \
  "$SAM2_HIERA_LARGE_CHECKPOINT" \
  "$SAM2_TINYVIT_21M_CHECKPOINT" \
  "$SAM2_TINYVIT_11M_CHECKPOINT" \
  "$SAM2_TINYVIT_5M_CHECKPOINT"; do
  test -f "$checkpoint" || echo "MISSING: $checkpoint"
done

sam2-trt list-models
```

先從 `sam2.1-hiera-tiny` 的 FP32 bundle 跑通，再展開其餘六個模型。

## 5. 在 Thor export 並建立 TensorRT bundle

設定共用 paths：

```bash
export SAM2_ROOT="$BENCH_ROOT/external/sam2"
export SAM2_DISTILL_ROOT="$BENCH_ROOT/external/SAM2-Distillation-Pipeline"
mkdir -p "$SAM2_TRT_ROOT/bundles" "$SAM2_TRT_ROOT/results/thor/engines"
```

### 5.1 第一個 accuracy-first FP32 bundle

```bash
cd "$SAM2_TRT_ROOT"

sam2-trt export \
  --model-id sam2.1-hiera-tiny \
  --sam2-root "$SAM2_ROOT" \
  --output-dir bundles/sam2.1-hiera-tiny/fp32 \
  --dtype fp32

sam2-trt build \
  --bundle-dir bundles/sam2.1-hiera-tiny/fp32 \
  --precision fp32 \
  --workspace-gib 8 \
  --builder-optimization-level 5 \
  --max-aux-streams 0

sam2-trt verify-bundle \
  --bundle-dir bundles/sam2.1-hiera-tiny/fp32
```

不要在 Thor 加 `--allow-non-thor`。成功後 bundle 應包含：

```text
encoder.onnx
prompt_point_step.onnx
prompt_box_step.onnx
track_step.onnx
encoder.fp32.engine
prompt_point_step.fp32.engine
prompt_box_step.fp32.engine
track_step.fp32.engine
manifest.json
build.json
timing.cache
```

### 5.2 TF32、FP16、BF16 candidates

每種 precision 使用獨立 bundle，避免 `manifest.json` 的 engine records 被下一次
build 覆寫：

```bash
# TF32 使用 FP32 ONNX export。
sam2-trt export \
  --model-id sam2.1-hiera-tiny \
  --sam2-root "$SAM2_ROOT" \
  --output-dir bundles/sam2.1-hiera-tiny/tf32 \
  --dtype fp32
sam2-trt build \
  --bundle-dir bundles/sam2.1-hiera-tiny/tf32 \
  --precision tf32

# FP16 使用 FP16 ONNX export。
sam2-trt export \
  --model-id sam2.1-hiera-tiny \
  --sam2-root "$SAM2_ROOT" \
  --output-dir bundles/sam2.1-hiera-tiny/fp16 \
  --dtype fp16
sam2-trt build \
  --bundle-dir bundles/sam2.1-hiera-tiny/fp16 \
  --precision fp16

# BF16 同理。
sam2-trt export \
  --model-id sam2.1-hiera-tiny \
  --sam2-root "$SAM2_ROOT" \
  --output-dir bundles/sam2.1-hiera-tiny/bf16 \
  --dtype bf16
sam2-trt build \
  --bundle-dir bundles/sam2.1-hiera-tiny/bf16 \
  --precision bf16
```

先以 FP32/no-TF32 作 accuracy reference，再比較 TF32、FP16、BF16。不要在沒有
real-input calibration 與 accuracy gate 的情況下直接開 FP8/INT8。

### 5.3 TinyViT 21M/11M/5M

以下以 21M 為例。最保守、一定與 checkpoint downstream weights 一致的做法是
完整 export 四張 graphs：

```bash
sam2-trt export \
  --model-id sam2.1-tinyvit-21m \
  --sam2-root "$SAM2_ROOT" \
  --distill-root "$SAM2_DISTILL_ROOT" \
  --output-dir bundles/sam2.1-tinyvit-21m/fp32 \
  --dtype fp32

sam2-trt build \
  --bundle-dir bundles/sam2.1-tinyvit-21m/fp32 \
  --precision fp32 \
  --builder-optimization-level 5 \
  --max-aux-streams 0
```

11M/5M 只需替換 `--model-id` 與 output directory。TinyViT model 仍使用
SAM2.1-L 的 prompt/mask/memory modules，因此必須設定
`SAM2_HIERA_LARGE_CHECKPOINT`。

Registry 會讓 5M/11M encoder 使用 Dynamo exporter，21M encoder 使用 legacy
exporter；這是 PACE 上避免 21M attention-bias cache 被展開成大型 ONNX graph 的
結果。四張 TensorRT plans 仍一律在 Thor 建立。

`--builder-optimization-level 5 --max-aux-streams 0` 是 L40S 的起始設定。完成
accuracy gate 後，另建三個乾淨 bundle，分別使用 `--max-aux-streams 0`、`1`、
`2` 比較；不要在同一 bundle 反覆 build，否則 manifest 與 timing cache 不容易
追溯。Thor 的勝者以完整 camera pipeline latency 為準。

只有在 distilled checkpoint 不含 `task_model_state`、且確認 downstream weights
就是相同 dtype 的 base SAM2.1-L 時，才使用 `--reuse-downstream-dir`。Exporter
會拒絕將 task-tuned downstream weights 錯誤替換成 base graphs。

## 6. Engine smoke benchmark

先以 synthetic tensors 檢查每張 engine 能 deserialize、選 profile 並執行：

```bash
cd "$SAM2_TRT_ROOT"
export BUNDLE="$SAM2_TRT_ROOT/bundles/sam2.1-hiera-tiny/fp32"
export ENGINE_RESULTS="$SAM2_TRT_ROOT/results/thor/engines/hiera-tiny-fp32"
mkdir -p "$ENGINE_RESULTS"

sam2-trt benchmark-engine \
  --engine "$BUNDLE/encoder.fp32.engine" \
  --role encoder --batch 1 --warmup 20 --runs 100 \
  --output "$ENGINE_RESULTS/encoder-b1.json"

for role in prompt_point_step prompt_box_step; do
  for batch in 1 2 4 8; do
    sam2-trt benchmark-engine \
      --engine "$BUNDLE/${role}.fp32.engine" \
      --role "$role" --batch "$batch" --warmup 20 --runs 100 \
      --output "$ENGINE_RESULTS/${role}-b${batch}.json"
  done
done

for batch in 1 2 4; do
  sam2-trt benchmark-engine \
    --engine "$BUNDLE/track_step.fp32.engine" \
    --role track_step --batch "$batch" --warmup 20 --runs 100 \
    --output "$ENGINE_RESULTS/track_step-b${batch}.json"
done
```

`track_step` 沒有 batch-8 profile。五到八個 objects 在 C++ runtime 中會拆成
最多兩個 batch-4 launches。不要對 `track_step` 傳 `--batch 8`。

Engine JSON 的 `mean_ms`/p50/p90/p99 是 preallocated graph execution latency，
不包含 image capture、ROS transport、host color conversion 或 display。

官方 Hiera models 可另外跑相同 graph 的 PyTorch comparison：

```bash
sam2-trt benchmark-pytorch-graphs \
  --model-id sam2.1-hiera-tiny \
  --sam2-root "$SAM2_ROOT" \
  --batch 1 --warmup 20 --runs 100 \
  --output "$ENGINE_RESULTS/pytorch-fp32-b1.json"
```

比較 TF32 bundle 時加 `--tf32`。此 command 目前只支援 official SAM2 encoders，
不支援 TinyViT rows。

## 7. Accuracy gate

PyTorch oracle 與 TensorRT candidate 必須使用：

- 同一 checkpoint SHA256；
- 同一 frames、prompt coordinates/object-add frames；
- 同一 mask threshold 與 postprocess；
- 同一 SA-V/SA1B or image manifest revision。

兩份 report 格式：

```json
{
  "metric_unit": "percentage_points",
  "metrics": {
    "sav_jf": 80.0,
    "image_miou": 75.0
  },
  "binary_masks_npz": "binary_masks.npz"
}
```

執行 gate：

```bash
sam2-trt validate \
  --baseline results/thor/accuracy/pytorch-fp32/report.json \
  --candidate results/thor/accuracy/tensorrt-fp32/report.json \
  --bundle-dir "$BUNDLE" \
  --maximum-metric-drop 0.1 \
  --minimum-frame-iou 0.999 \
  --output results/thor/accuracy/tensorrt-fp32/gate.json
```

Exit 0 才通過；exit 2 表示拒絕 candidate。0.1 是 percentage-point drop，不是
相對百分比。

目前 repo 尚未提供從 TensorRT runtime 自動產生 candidate report/NPZ 的 command。
在該 runner 完成前，engine benchmark 與 ROS smoke 都不能取代 accuracy gate。

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
{"stamp_ns":123456789,"frame_index":42,"objects":[1,2],"queue_wait_ms":0.031,"color_convert_ms":0.481,"inference_ms":21.732,"mask_publish_ms":0.109,"callback_total_ms":22.368,"frame_interval_ms":33.333,"tracking_fps":30.000,"dropped":0,"dropped_frames":17,"source_age_ms":24.126,"end_to_end_ms":24.126}
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

目前 node 不含 interactive click UI。先從 image viewer 讀出 pixel coordinates，再用
service 加 object。

## 12. 效能記錄方式

### 12.1 ROS camera pipeline JSONL

Launch 時傳入 `trace_path` 後，node 會將與 `/sam/result_json` 相同的每-frame
JSON append 到該檔案。請為每次實驗使用新的檔名，避免 append 混入舊 run：

```bash
ros2 launch sam2_trt_ros camera_stream.launch.py \
  bundle_dir:="$SAM2_TRT_ROOT/bundles/sam2.1-hiera-tiny/fp32" \
  precision:=fp32 \
  image_topic:=/image \
  trace_path:="$SAM2_TRT_ROOT/results/thor/run_001/runtime.jsonl"

sam2-trt benchmark \
  --trace "$SAM2_TRT_ROOT/results/thor/run_001/runtime.jsonl" \
  --output "$SAM2_TRT_ROOT/results/thor/run_001/runtime_summary.json"
```

摘要包含存在於 trace 中的 `queue_wait_ms`、`color_convert_ms`、`inference_ms`、
`mask_publish_ms`、`callback_total_ms`、`source_age_ms`、`end_to_end_ms`、
`tracking_fps` 的 mean/p50/p90/p99，以及總 throughput 與 dropped frames。

`inference_ms` 是 `Tracker::process_rgb8` 的完整 wall time，包含 CUDA/TensorRT
執行及 runtime 為回傳 mask 所需的同步；它不是單獨 engine kernel time。
`callback_total_ms` 從 ROS subscription 收到 frame 算到 result publish 前；
`end_to_end_ms` 則使用 image header timestamp，因此只有 timestamp clock 正確時才可信。

### 12.2 其他應一起記錄

- `benchmark-engine` 的 mean/p50/p90/p99 與 object throughput；
- camera publish FPS：`ros2 topic hz <image_topic>`；
- mask/result publish FPS：`ros2 topic hz /sam/result_json`；
- `/sam/result_json` 的累計 dropped frames；
- `tegrastats` 的 power、temperature、memory 與 utilization；
- model/checkpoint/engine hashes、precision、power mode、clocks、camera profile。

```bash
mkdir -p "$SAM2_TRT_ROOT/logs/thor"
tegrastats --interval 1000 | tee "$SAM2_TRT_ROOT/logs/thor/tegrastats.log"
```

停止時用 `Ctrl-C`。

### 12.3 目前不可由 ROS node 直接宣稱

- preprocess/encoder/tail/postprocess 分段 latency；
- overlay/display FPS；
- TensorRT-vs-PyTorch real-input mask parity。

不要把 `inference_ms` 誤標成 encoder-only latency，也不要在 input timestamp 為零或
不同 clock domain 時把缺少的 `end_to_end_ms` 補成猜測值。

## 13. 建議測試矩陣

每個 model/precision 至少跑：

| Test | Objects | Prompt | Source | 驗證內容 |
| --- | ---: | --- | --- | --- |
| Engine smoke | 1 | point | synthetic | 四張 engines 可執行 |
| Prompt batching | 1/2/4/8 | point、box | synthetic | profiles 與 throughput |
| Track batching | 1/2/4 | memory | synthetic | profiles 與 memory shapes |
| ROS video smoke | 1 | point | `videos/test1.mov` | service、mask、result topic |
| ROS video multi-object | 2/4/8 | point + box | `videos/test2.mov` | IDs、batch split、reset |
| RealSense smoke | 1 | point | actual camera | QoS、encoding、latest-frame behavior |
| RealSense multi-object | 2/4/8 | point + box | actual camera | tracking stability、dropped frames |
| Accuracy | dataset-defined | fixed | SA-V/SA1B/images | J&F、mIoU、per-mask IoU |

Precision promotion 順序：

```text
FP32/no-TF32 correctness
  -> TF32 accuracy + speed
  -> FP16 accuracy + speed
  -> BF16 accuracy + speed
  -> only then consider calibrated lower precision
```

## 14. 每次 run 要保存的紀錄

```text
date/time
Thor hostname and device model
JetPack/L4T, CUDA, TensorRT, PyTorch, ROS versions
Efficient-SAM2-TensorRT commit
SAM2 and distillation source commits
model ID
encoder and downstream checkpoint paths + SHA256
bundle path + manifest.json
precision
power mode and jetson_clocks state
camera/video source, resolution, source FPS and ROS topic
object count and point/box coordinates
engine benchmark JSON paths
rosbag/result logs
mask/result FPS and dropped-frame count
tegrastats log
accuracy report and gate result when available
```

Bundles、engines、checkpoints、rosbags、results、logs 與 videos 都是 local artifacts，
不要提交 Git。

## 15. Troubleshooting

### `sam2-trt pin` 說不是 Thor

查看：

```bash
tr -d '\0' </proc/device-tree/model
cat results/thor/environment_probe.json
```

不要用 `--allow-non-thor` 繞過正式 Thor build。

### `import tensorrt` 失敗或 engine deserialize 失敗

確認 venv 是 `--system-site-packages`，Python binding、headers、runtime library 來自
同一 JetPack。Engine 必須在目前 Thor、目前 TensorRT stack 重建；不要搬用
PACE/H100/L40S plans。

### CMake 找不到 `NvInfer.h` 或 `libnvinfer.so`

用 `dpkg -L` 與 `ldconfig -p | grep nvinfer` 找 JetPack 實際位置，再傳
`TENSORRT_INCLUDE_DIR`/`TENSORRT_LIBRARY`。不要下載不同 major version library
硬接。

### `cv_bridge` 出現 NumPy ABI error

```bash
python -m pip install --force-reinstall "numpy>=1.26,<2"
```

再重新 source environment。

### `ros2` 找不到 package/executable/service type

```bash
source /opt/ros/jazzy/setup.bash
source "$SAM2_TRT_ROOT/ros_ws/install/setup.bash"
export LD_LIBRARY_PATH="$SAM2_TRT_ROOT/build/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ros2 pkg executables sam2_trt_ros
ros2 interface show sam2_trt_msgs/srv/AddObject
```

修改 C++/ROS source 後必須重新 `cmake --build`、`cmake --install`、
`colcon build` 並重新 source。

### Camera 有 topic，但 TensorRT node 沒有輸出

確認：

- image encoding 是 `rgb8` 或 `bgr8`；
- publisher 與 subscriber 使用相同 `ROS_DOMAIN_ID`；
- RealSense topic 是實際存在的 `/camera/camera/color/image_raw`；
- engine precision filename 與 launch 的 `precision` 相同；
- Terminal B 可以讀取 bundle 並載入 `libnvinfer.so`。

### Masks 空白或 prompt 在錯誤位置

Service coordinates 是原始 frame pixels。先查 `camera_info` width/height，從中央
point 開始，再以 `rqt_image_view` 檢查。速度快但 mask 錯誤不算成功。

### Camera FPS 不符合設定

以 `ros2 param get`、`camera_info`、`ros2 topic hz` 的實際結果為準。若
`lsusb -t` 顯示 `480M`，改用直接連接的 USB 3 port/cable。

### 八物件 track engine profile error

不要直接要求 track batch 8。目前 runtime 應自動拆成兩個 batch 4；若仍看到
profile 3 或 batch-8 track request，表示 Thor checkout/ROS install 不是最新版，
請重新 build 並 source。

## 16. Thor acceptance checklist

只有全部勾選後才算完成 Thor deployment：

- [ ] Environment probe 與 lock 已保存。
- [ ] Checkpoint/source commits 與 SHA256 已保存。
- [ ] 四張 FP32 engines 都在該 Thor 建立且 `verify-bundle` 通過。
- [ ] Engine smoke 與 1/2/4/8 prompt profiles 通過。
- [ ] Track 1/2/4 profiles 通過；八物件 runtime split 通過。
- [ ] C++ unit test 與 ROS workspace build 通過。
- [ ] Recorded-video point、box、multi-object、reset smoke 通過。
- [ ] RealSense negotiated profile、USB speed 與 source FPS 已確認。
- [ ] RealSense masks、IDs、latest-frame dropped behavior 已確認。
- [ ] FP32 real-input accuracy report 與 gate 通過。
- [ ] Candidate precision 的 accuracy gate 通過後才比較/採用速度。
- [ ] JSONL trace 已彙整，source timestamp clock 已確認後才宣稱 end-to-end latency。
- [ ] 若需宣稱 encoder/tail/postprocess 分段，先補 core-level instrumentation。
