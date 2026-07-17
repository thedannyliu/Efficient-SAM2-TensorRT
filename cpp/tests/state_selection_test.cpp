#include "sam2_trt/state.hpp"

#include <cassert>
#include <map>

int main() {
  std::map<int, int> conditioning{{0, 100}};
  std::map<int, int> non_conditioning;
  for (int frame = 1; frame < 20; ++frame) non_conditioning[frame] = 100 + frame;

  const auto selected = sam2_trt::select_state(20, 21, conditioning, non_conditioning);
  assert(selected.memories.size() == 7);
  assert(selected.memories.front().frame == 0);
  assert(selected.memories.back().frame == 19);
  assert(selected.pointers.size() == 16);
  assert(selected.pointers.front().frame == 0);
  assert(selected.pointers.back().frame == 5);
  assert(sam2_trt::padded_object_batch(3) == 4);

  std::map<int, int> many_conditioning{{0, 0}, {4, 4}, {8, 8}, {12, 12}};
  const auto [closest, remaining] =
      sam2_trt::select_closest_conditioning(10, many_conditioning, 2);
  assert(closest.contains(8));
  assert(closest.contains(12));
  assert(remaining.size() == 2);
  return 0;
}
