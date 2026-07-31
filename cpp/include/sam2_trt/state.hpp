#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sam2_trt {

inline int floor_div(int numerator, int denominator) {
  int quotient = numerator / denominator;
  const int remainder = numerator % denominator;
  if (remainder != 0 && ((remainder < 0) != (denominator < 0))) --quotient;
  return quotient;
}

template <class T> struct SelectedState {
  struct Item {
    int position;
    int frame;
    T value;
  };
  std::vector<Item> memories;
  std::vector<Item> pointers;
};

template <class T>
std::pair<std::map<int, T>, std::map<int, T>> select_closest_conditioning(
    int frame, const std::map<int, T>& conditioning, int maximum) {
  if (maximum == -1 || static_cast<int>(conditioning.size()) <= maximum) {
    return {conditioning, {}};
  }
  if (maximum < 2) throw std::invalid_argument("maximum conditioning frames must be -1 or >= 2");
  std::map<int, T> selected;
  auto after = conditioning.lower_bound(frame);
  if (after != conditioning.end()) selected.insert(*after);
  if (after != conditioning.begin()) selected.insert(*std::prev(after));
  std::vector<typename std::map<int, T>::const_iterator> candidates;
  for (auto it = conditioning.begin(); it != conditioning.end(); ++it) {
    if (!selected.contains(it->first)) candidates.push_back(it);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [frame](auto lhs, auto rhs) {
    return std::abs(lhs->first - frame) < std::abs(rhs->first - frame);
  });
  for (auto it : candidates) {
    if (static_cast<int>(selected.size()) == maximum) break;
    selected.insert(*it);
  }
  std::map<int, T> unselected;
  for (const auto& item : conditioning) if (!selected.contains(item.first)) unselected.insert(item);
  return {selected, unselected};
}

template <class T>
SelectedState<T> select_state(
    int frame, int frame_count, const std::map<int, T>& conditioning,
    const std::map<int, T>& non_conditioning, int num_maskmem = 7,
    int max_conditioning = -1, int stride = 1, int max_pointers = 16) {
  if (num_maskmem < 1 || stride < 1 || max_pointers < 1) throw std::invalid_argument("invalid state limits");
  auto [selected, unselected] = select_closest_conditioning(frame, conditioning, max_conditioning);
  SelectedState<T> result;
  for (const auto& [index, value] : selected) result.memories.push_back({0, index, value});
  for (int temporal = 1; temporal < num_maskmem; ++temporal) {
    const int relative = num_maskmem - temporal;
    const int index = relative == 1
        ? frame - 1
        : floor_div(frame - 2, stride) * stride - (relative - 2) * stride;
    auto it = non_conditioning.find(index);
    if (it != non_conditioning.end()) result.memories.push_back({temporal, index, it->second});
    else if (auto cond = unselected.find(index); cond != unselected.end())
      result.memories.push_back({temporal, index, cond->second});
  }
  for (const auto& [index, value] : selected)
    if (index <= frame) result.pointers.push_back({frame - index, index, value});
  const int maximum = std::min(frame_count, max_pointers);
  for (int distance = 1; distance < maximum; ++distance) {
    const int index = frame - distance;
    if (index < 0) break;
    auto it = non_conditioning.find(index);
    if (it != non_conditioning.end()) result.pointers.push_back({distance, index, it->second});
    else if (auto cond = unselected.find(index); cond != unselected.end())
      result.pointers.push_back({distance, index, cond->second});
  }
  return result;
}

inline int padded_object_batch(int count) {
  for (int batch : {1, 2, 4, 8}) if (count <= batch && count > 0) return batch;
  throw std::invalid_argument("object count must be in [1, 8]");
}

inline std::vector<int> track_bucket_group_sizes(
    int compatible_count, int object_count, int bucket_size,
    int minimum_objects) {
  if (compatible_count < 0 || object_count < compatible_count)
    throw std::invalid_argument("invalid compatible object count");
  if (bucket_size != 1 && bucket_size != 2 && bucket_size != 4)
    throw std::invalid_argument("track bucket size must be 1, 2, or 4");
  if (minimum_objects < 1)
    throw std::invalid_argument("minimum bucket objects must be positive");
  const int capacity =
      bucket_size > 1 && object_count >= minimum_objects ? bucket_size : 1;
  std::vector<int> result;
  for (int remaining = compatible_count; remaining > 0; remaining -= capacity)
    result.push_back(std::min(remaining, capacity));
  return result;
}

inline int track_bucket_for_object_count(
    int object_count, int fixed_bucket_size, int minimum_objects,
    const std::vector<int>& router = {}) {
  if (object_count < 0)
    throw std::invalid_argument("object count must be non-negative");
  if (fixed_bucket_size != 1 && fixed_bucket_size != 2 &&
      fixed_bucket_size != 4)
    throw std::invalid_argument("track bucket size must be 1, 2, or 4");
  if (minimum_objects < 1)
    throw std::invalid_argument("minimum bucket objects must be positive");
  for (const int bucket : router)
    if (bucket != 1 && bucket != 2 && bucket != 4)
      throw std::invalid_argument(
          "track bucket router values must be 1, 2, or 4");
  if (object_count == 0) return 1;
  if (!router.empty()) {
    if (object_count > static_cast<int>(router.size()))
      throw std::invalid_argument(
          "track bucket router does not cover the object count");
    return router[static_cast<std::size_t>(object_count - 1)];
  }
  return fixed_bucket_size > 1 && object_count >= minimum_objects
      ? fixed_bucket_size
      : 1;
}

}  // namespace sam2_trt
