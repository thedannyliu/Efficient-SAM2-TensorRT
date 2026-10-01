# Measurement protocol

[Guide index](../thor_testing_guide.md) · [Runtime architecture](../architecture.md)

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
`mask_publish_ms`、`worker_total_ms`、`callback_total_ms`、`source_age_ms`、`end_to_end_ms`、
`processed_fps` 的 mean/p50/p90/p99，以及總 throughput、measurement duration
與 dropped frames。`tracking_fps` 是保留給舊 trace 的 `processed_fps` alias。

`inference_ms` 是 `Tracker::process_rgb8` 的完整 wall time，包含 CUDA/TensorRT
執行及 runtime 為回傳 mask 所需的同步；它不是單獨 engine kernel time。
`worker_total_ms` 從 worker 取出 frame 算到 mask publish 完成，不含排隊；
`callback_total_ms` 從 ROS subscription 收到 frame 算到 mask publish 完成，包含排隊；
`end_to_end_ms` 則使用 image header timestamp，因此只有 timestamp clock 正確時才可信。

不要預期 `1000 / inference_ms` 等於 camera FPS。前者接近 pipeline 在 input
持續供應時的 service capacity；後者受 camera publish cadence、USB、queue 與
dropped frames 限制。node 另外輸出 `processing_capacity_fps`（單幀
`1000 / worker_total_ms`，不含 queue wait）與 `processed_fps`（相鄰 processed frame start
interval）。正式 run 的實際 FPS 使用 summary 的 `throughput_fps`：
`interval_count / measurement_duration_s`。不要對逐幀 `processed_fps` 做算術
平均來代替 throughput，因為長 frame gap 會被錯誤低估。

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
