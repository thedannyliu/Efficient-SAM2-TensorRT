# Export and accuracy validation

[Guide index](../thor_testing_guide.md) · [Runtime architecture](../architecture.md)

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
