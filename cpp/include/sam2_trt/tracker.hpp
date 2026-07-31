#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sam2_trt {

enum class PromptKind { Point, Box, Mask };

struct Prompt {
  PromptKind kind{PromptKind::Point};
  float x0{};
  float y0{};
  float x1{};
  float y1{};
  int mask_width{};
  int mask_height{};
  std::size_t mask_stride{};
  std::vector<std::uint8_t> mask;
};

struct ObjectMask {
  int object_id{};
  int width{};
  int height{};
  std::vector<std::uint8_t> mono8;
};

struct TrackerTimings {
  double host_input_copy_ms{};
  double encoder_gpu_ms{};
  double tail_gpu_ms{};
  double gpu_total_ms{};
  double host_mask_copy_ms{};
  double total_ms{};
};

class Tracker {
 public:
  Tracker(
      const std::string& bundle_directory, const std::string& precision,
      int max_objects = 8, int track_concurrency = 8,
      int track_bucket_size = 1, int track_bucket_min_objects = 4,
      bool fused_state_gather = false,
      std::vector<int> track_bucket_router = {});
  ~Tracker();
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  int add_object(const Prompt& prompt);
  void reset();
  std::vector<ObjectMask> process_rgb8(
      const std::uint8_t* image, int width, int height, std::size_t row_stride);
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
