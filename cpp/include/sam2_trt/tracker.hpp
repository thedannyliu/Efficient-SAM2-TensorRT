#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sam2_trt {

enum class PromptKind { Point, Box };

// Prompt coordinates are expressed in source-frame pixels.
struct Prompt {
  PromptKind kind{PromptKind::Point};
  float x0{};
  float y0{};
  float x1{};
  float y1{};
};

// One object mask copied back to host memory for ROS publication.
struct ObjectMask {
  int object_id{};
  int width{};
  int height{};
  std::vector<std::uint8_t> mono8;
};

// Timings separate data movement, encoder work, tracking tail, and wall time.
struct TrackerTimings {
  double host_input_copy_ms{};
  double encoder_gpu_ms{};
  double tail_gpu_ms{};
  double gpu_total_ms{};
  double host_mask_copy_ms{};
  double total_ms{};
};

// Stateful SAM2 inference facade backed by four TensorRT engines:
// encoder, point prompt, box prompt, and per-frame tracking.
class Tracker {
 public:
  Tracker(
      const std::string& bundle_directory, const std::string& precision,
      int max_objects = 8, int track_concurrency = 8,
      int track_bucket_size = 1, int track_bucket_min_objects = 4);
  ~Tracker();
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  // Queue a prompt; initialization occurs against the next processed frame.
  int add_object(const Prompt& prompt);
  void reset();
  std::vector<ObjectMask> process_rgb8(
      const std::uint8_t* image, int width, int height, std::size_t row_stride);
  // Overlap the current encoder with tracking of the previous encoded frame.
  // The first call intentionally returns no output while filling the pipeline.
  std::optional<std::vector<ObjectMask>> process_pipelined_rgb8(
      const std::uint8_t* image, int width, int height,
      std::size_t row_stride);
  void discard_pipelined_frame();
  TrackerTimings last_timings() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sam2_trt
