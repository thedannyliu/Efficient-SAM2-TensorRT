# Runtime architecture

The runtime has four graph roles: encoder, point initialization, box
initialization and temporal tracking. Engine bundles bind them to checkpoint
hashes, tensor shapes and precision settings.

## Frame and state ownership

1. The ROS node receives a frame and applies the latest-frame policy.
2. `cpp/src/tracker.cpp` stages RGB bytes in pinned memory and runs the encoder.
3. Newly prompted objects run a point or box graph; existing objects select
   their conditioning and temporal history and run the tracking graph.
4. CUDA kernels convert results; the tracker retains memory features and object
   pointers for the next frame, while ROS publishes masks and timing metadata.

`cpp/src/engine.cpp` owns TensorRT execution contexts. `cpp/src/kernels.cu`
owns device transforms; `cpp/include/sam2_trt/state.hpp` implements history
selection. The Python reference in `sam2_trt/state.py` and the C++ state test
exercise the same ordering rules.

Objects may share an encoder result but must retain independent temporal state.
Optional bucket execution groups compatible memory/pointer shapes. Optional
cross-frame overlap deliberately delays results by one frame; report source age
alongside throughput before enabling it for a camera workflow.

## Component boundaries

| Change | Primary location |
| --- | --- |
| Export contract / model reconstruction | `sam2_trt/export.py`, `model_registry.py` |
| Engine build and profiles | `sam2_trt/build.py`, `manifest.py` |
| Runtime scheduling and buffers | `cpp/src/tracker.cpp` |
| Model switching, ROS services and timestamps | `ros_ws/src/sam2_trt_ros/` |
| Accuracy acceptance | `sam2_trt/validate.py`, `prompt_parity.py` |

The companion SAM3 integration can initialize SAM2 objects from text detections.
That orchestration and its proprietary detector runtime are separate from this
repository's SAM2 tracker.

## Artifact changes

Create a new bundle directory and manifest for a new engine/checkpoint pair.
Do not overwrite an engine in an existing validated bundle. Rebuild on the
target GPU, validate the exact bundle, and preserve its environment and hashes.
The supplied Thor handoff contains LFS pointers for engines and weights;
those pointers are not runnable artifacts or a hardware-validation result.
