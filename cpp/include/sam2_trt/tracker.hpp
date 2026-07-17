#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sam2_trt {

enum class PromptKind { Point, Box };

struct Prompt {
  PromptKind kind{PromptKind::Point};
  float x0{};
  float y0{};
  float x1{};
  float y1{};
};

struct ObjectMask {
  int object_id{};
  int width{};
  int height{};
  std::vector<std::uint8_t> mono8;
};

class Tracker {
 public:
  Tracker(const std::string& bundle_directory, const std::string& precision, int max_objects = 8);
  ~Tracker();
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  int add_object(const Prompt& prompt);
  void reset();
  std::vector<ObjectMask> process_rgb8(
      const std::uint8_t* image, int width, int height, std::size_t row_stride);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sam2_trt
