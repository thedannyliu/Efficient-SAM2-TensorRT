# Troubleshooting and acceptance

[Guide index](../thor_testing_guide.md) · [Runtime architecture](../architecture.md)

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
