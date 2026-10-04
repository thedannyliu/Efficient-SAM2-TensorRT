# Environment and model inputs

[Guide index](../thor_testing_guide.md) · [Runtime architecture](../architecture.md)

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
