#include "sam2_trt/tracker.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::map<int, const sam2_trt::ObjectMask*> by_id(
    const std::vector<sam2_trt::ObjectMask>& masks) {
  std::map<int, const sam2_trt::ObjectMask*> result;
  for (const auto& mask : masks) result.emplace(mask.object_id, &mask);
  return result;
}

double average(const std::vector<double>& values) {
  if (values.empty()) throw std::invalid_argument("no measured values");
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) {
    std::cerr
        << "usage: tracker_fused_gather_ab BUNDLE [OBJECTS] [WARMUP] [RUNS]\n";
    return 2;
  }
  const std::string bundle = argv[1];
  const int objects = argc > 2 ? std::atoi(argv[2]) : 4;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
  const int runs = argc > 4 ? std::atoi(argv[4]) : 100;
  if (objects < 1 || objects > 8 || warmup < 16 || runs < 1) {
    std::cerr << "invalid objects, warmup, or runs\n";
    return 2;
  }

  constexpr int width = 848;
  constexpr int height = 480;
  constexpr int channels = 3;
  const int concurrency = std::min(objects, 4);
  sam2_trt::Tracker baseline(
      bundle, "fp16", 8, concurrency, 1, 4, false);
  sam2_trt::Tracker candidate(
      bundle, "fp16", 8, concurrency, 1, 4, true);
  for (int object = 0; object < objects; ++object) {
    const float fraction =
        static_cast<float>(object + 1) / static_cast<float>(objects + 1);
    sam2_trt::Prompt prompt;
    prompt.x0 = fraction * width;
    prompt.y0 = (0.35f + 0.3f * (object % 2)) * height;
    baseline.add_object(prompt);
    candidate.add_object(prompt);
  }

  std::vector<std::uint8_t> image(width * height * channels);
  std::vector<double> baseline_ms;
  std::vector<double> candidate_ms;
  std::uint64_t equal_pixels = 0;
  std::uint64_t total_pixels = 0;
  std::uint64_t intersection = 0;
  std::uint64_t union_count = 0;
  const int total_frames = warmup + runs;
  for (int frame = 0; frame < total_frames; ++frame) {
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int channel = 0; channel < channels; ++channel)
          image[(y * width + x) * channels + channel] =
              static_cast<std::uint8_t>(
                  (x + 2 * y + 13 * channel + 3 * frame) % 256);

    std::vector<sam2_trt::ObjectMask> baseline_masks;
    std::vector<sam2_trt::ObjectMask> candidate_masks;
    if (frame % 2 == 0) {
      baseline_masks =
          baseline.process_rgb8(image.data(), width, height, width * channels);
      candidate_masks =
          candidate.process_rgb8(image.data(), width, height, width * channels);
    } else {
      candidate_masks =
          candidate.process_rgb8(image.data(), width, height, width * channels);
      baseline_masks =
          baseline.process_rgb8(image.data(), width, height, width * channels);
    }
    if (frame >= warmup) {
      baseline_ms.push_back(baseline.last_timings().total_ms);
      candidate_ms.push_back(candidate.last_timings().total_ms);
    }

    const auto baseline_by_id = by_id(baseline_masks);
    const auto candidate_by_id = by_id(candidate_masks);
    if (baseline_by_id.size() != candidate_by_id.size())
      throw std::runtime_error("tracker object counts differ");
    for (const auto& [id, reference] : baseline_by_id) {
      const auto found = candidate_by_id.find(id);
      if (found == candidate_by_id.end())
        throw std::runtime_error("tracker object IDs differ");
      const auto& value = *found->second;
      if (reference->mono8.size() != value.mono8.size())
        throw std::runtime_error("tracker mask dimensions differ");
      for (std::size_t index = 0; index < reference->mono8.size(); ++index) {
        const bool a = reference->mono8[index] != 0;
        const bool b = value.mono8[index] != 0;
        equal_pixels += a == b;
        intersection += a && b;
        union_count += a || b;
        ++total_pixels;
      }
    }
  }

  const double baseline_mean = average(baseline_ms);
  const double candidate_mean = average(candidate_ms);
  const double binary_iou =
      union_count == 0
          ? 1.0
          : static_cast<double>(intersection) / union_count;
  std::cout << "{\n"
            << "  \"objects\": " << objects << ",\n"
            << "  \"warmup\": " << warmup << ",\n"
            << "  \"runs\": " << runs << ",\n"
            << "  \"baseline_mean_ms\": " << baseline_mean << ",\n"
            << "  \"candidate_mean_ms\": " << candidate_mean << ",\n"
            << "  \"speedup\": " << baseline_mean / candidate_mean << ",\n"
            << "  \"equal_pixel_fraction\": "
            << static_cast<double>(equal_pixels) / total_pixels << ",\n"
            << "  \"binary_iou\": " << binary_iou << "\n"
            << "}\n";
  return 0;
}
