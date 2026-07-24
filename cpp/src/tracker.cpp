#include "sam2_trt/tracker.hpp"

#include "sam2_trt/engine.hpp"
#include "sam2_trt/kernels.hpp"
#include "sam2_trt/state.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace sam2_trt {
namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

DeviceTensor slice_batch(const DeviceTensor& tensor, int index) {
  if (tensor.shape.empty() || index >= tensor.shape.front()) throw std::out_of_range("batch slice");
  auto shape = tensor.shape;
  shape[0] = 1;
  std::size_t bytes = tensor.bytes / static_cast<std::size_t>(tensor.shape.front());
  return {tensor.storage, static_cast<std::byte*>(tensor.data) + index * bytes, std::move(shape), tensor.dtype, bytes};
}

DeviceTensor repeat_batch(const DeviceTensor& source, int batch, cudaStream_t stream) {
  if (!source.shape.empty() && source.shape.front() == batch) return source;
  auto shape = source.shape;
  shape[0] = batch;
  auto output = allocate_tensor(shape, source.dtype, stream);
  const std::size_t per_batch = source.bytes / static_cast<std::size_t>(source.shape.front());
  for (int index = 0; index < batch; ++index)
    check_cuda(cudaMemcpyAsync(static_cast<std::byte*>(output.data) + index * per_batch,
                               source.data, per_batch, cudaMemcpyDeviceToDevice, stream));
  return output;
}

class PinnedBytes {
 public:
  PinnedBytes() = default;
  ~PinnedBytes() {
    if (data_) cudaFreeHost(data_);
  }
  PinnedBytes(const PinnedBytes&) = delete;
  PinnedBytes& operator=(const PinnedBytes&) = delete;

  std::uint8_t* ensure(std::size_t bytes) {
    if (bytes <= capacity_) return data_;
    if (data_) check_cuda(cudaFreeHost(data_));
    check_cuda(cudaHostAlloc(
        reinterpret_cast<void**>(&data_), bytes, cudaHostAllocPortable));
    capacity_ = bytes;
    return data_;
  }

 private:
  std::uint8_t* data_{};
  std::size_t capacity_{};
};

int profile_for_batch(int batch) {
  switch (batch) { case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3; }
  throw std::invalid_argument("unsupported object batch");
}

}  // namespace

struct Tracker::Impl {
  using Clock = std::chrono::steady_clock;
  struct FrameState {
    DeviceTensor memory;
    DeviceTensor memory_position;
    DeviceTensor pointer;
  };
  struct ObjectState {
    int id;
    Prompt prompt;
    bool pending{true};
    std::map<int, std::shared_ptr<FrameState>> conditioning;
    std::map<int, std::shared_ptr<FrameState>> non_conditioning;
  };

  Engine encoder;
  Engine point_prompt;
  Engine box_prompt;
  std::vector<std::unique_ptr<Engine>> track_engines;
  cudaStream_t stream{};
  std::array<cudaStream_t, 8> track_streams{};
  cudaEvent_t encoded_ready{};
  cudaEvent_t gpu_start{};
  cudaEvent_t encoder_end{};
  cudaEvent_t gpu_end{};
  int maximum_objects;
  int track_concurrency;
  int next_id{1};
  int frame_index{0};
  std::vector<ObjectState> objects;
  std::array<PinnedBytes, 2> input_staging;
  std::array<PinnedBytes, 8> mask_staging;
  std::size_t input_staging_index{};
  TrackerTimings timings;
  std::mutex timing_mutex;
  std::mutex mutex;

  Impl(
      const std::string& root, const std::string& precision, int maximum,
      int concurrency)
      : encoder((std::filesystem::path(root) / ("encoder." + precision + ".engine")).string()),
        point_prompt((std::filesystem::path(root) / ("prompt_point_step." + precision + ".engine")).string()),
        box_prompt((std::filesystem::path(root) / ("prompt_box_step." + precision + ".engine")).string()),
        maximum_objects(maximum),
        track_concurrency(concurrency) {
    if (maximum < 1 || maximum > 8) throw std::invalid_argument("max_objects must be in [1, 8]");
    if (concurrency < 1 || concurrency > maximum)
      throw std::invalid_argument("track_concurrency must be in [1, max_objects]");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check_cuda(cudaEventCreateWithFlags(&encoded_ready, cudaEventDisableTiming));
    const auto track_path =
        (std::filesystem::path(root) / ("track_step." + precision + ".engine")).string();
    track_engines.reserve(static_cast<std::size_t>(maximum));
    for (int index = 0; index < maximum; ++index) {
      check_cuda(cudaStreamCreateWithFlags(
          &track_streams[static_cast<std::size_t>(index)],
          cudaStreamNonBlocking));
      track_engines.push_back(std::make_unique<Engine>(track_path, true));
    }
    check_cuda(cudaEventCreate(&gpu_start));
    check_cuda(cudaEventCreate(&encoder_end));
    check_cuda(cudaEventCreate(&gpu_end));
    int device = 0;
    check_cuda(cudaGetDevice(&device));
    cudaMemPool_t pool{};
    check_cuda(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
    check_cuda(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  }

  ~Impl() {
    objects.clear();
    cudaStreamSynchronize(stream);
    for (int index = 0; index < maximum_objects; ++index) {
      auto object_stream = track_streams[static_cast<std::size_t>(index)];
      cudaStreamSynchronize(object_stream);
      cudaStreamDestroy(object_stream);
    }
    track_engines.clear();
    cudaEventDestroy(encoded_ready);
    cudaEventDestroy(gpu_end);
    cudaEventDestroy(encoder_end);
    cudaEventDestroy(gpu_start);
    cudaStreamDestroy(stream);
  }

  std::map<std::string, DeviceTensor> encode(
      const std::uint8_t* host_image, int width, int height, std::size_t stride) {
    const std::size_t input_bytes = stride * static_cast<std::size_t>(height);
    auto* pinned = input_staging[input_staging_index++ % input_staging.size()].ensure(
        input_bytes);
    const auto copy_start = Clock::now();
    std::memcpy(pinned, host_image, input_bytes);
    timings.host_input_copy_ms += std::chrono::duration<double, std::milli>(
        Clock::now() - copy_start).count();
    auto device_image = allocate_tensor(
        {height, static_cast<int64_t>(stride)}, nvinfer1::DataType::kUINT8, stream);
    check_cuda(cudaMemcpy2DAsync(device_image.data, stride, pinned, stride, stride, height,
                                 cudaMemcpyHostToDevice, stream));
    const auto dtype = encoder.tensor_dtype("image");
    auto normalized = allocate_tensor({1, 3, 1024, 1024}, dtype, stream);
    launch_preprocess_rgb8(static_cast<const std::uint8_t*>(device_image.data), width, height,
                           stride, normalized.data, dtype, stream);
    return encoder.run({{"image", normalized}}, 0, stream);
  }

  std::map<std::string, DeviceTensor> common_features(
      const std::map<std::string, DeviceTensor>& encoded,
      int batch,
      cudaStream_t execution_stream) {
    return {
      {"high_res_s0", repeat_batch(
          encoded.at("high_res_s0"), batch, execution_stream)},
      {"high_res_s1", repeat_batch(
          encoded.at("high_res_s1"), batch, execution_stream)},
      {"image_embedding", repeat_batch(
          encoded.at("image_embedding"), batch, execution_stream)},
    };
  }

  DeviceTensor device_from_i64(
      const std::vector<int64_t>& values,
      std::vector<int64_t> shape,
      cudaStream_t execution_stream) {
    auto tensor = allocate_tensor(
        std::move(shape), nvinfer1::DataType::kINT64, execution_stream);
    check_cuda(cudaMemcpyAsync(
        tensor.data,
        values.data(),
        tensor.bytes,
        cudaMemcpyHostToDevice,
        execution_stream));
    return tensor;
  }

  void save_outputs(ObjectState& object, const std::map<std::string, DeviceTensor>& output,
                    int batch_index, bool conditioning) {
    auto state = std::make_shared<FrameState>(FrameState{
      slice_batch(output.at("new_memory"), batch_index),
      slice_batch(output.at("new_memory_position"), batch_index),
      slice_batch(output.at("object_pointer"), batch_index),
    });
    (conditioning ? object.conditioning : object.non_conditioning)[frame_index] = std::move(state);
    while (object.non_conditioning.size() > 16) object.non_conditioning.erase(object.non_conditioning.begin());
  }

  std::vector<ObjectMask> masks_from_output(
      const std::vector<ObjectState*>& group, const std::map<std::string, DeviceTensor>& output,
      int width, int height, cudaStream_t execution_stream) {
    const auto& logits = output.at("mask_logits");
    std::vector<ObjectMask> result;
    result.reserve(group.size());
    const std::size_t per_batch = logits.bytes / static_cast<std::size_t>(logits.shape.front());
    for (std::size_t index = 0; index < group.size(); ++index) {
      auto mono = allocate_tensor(
          {height, width}, nvinfer1::DataType::kUINT8, execution_stream);
      launch_mask_to_mono8(static_cast<const std::byte*>(logits.data) + index * per_batch,
                           logits.dtype, 1024, 1024, static_cast<std::uint8_t*>(mono.data),
                           width, height, execution_stream);
      ObjectMask mask{group[index]->id, width, height, std::vector<std::uint8_t>(width * height)};
      auto* pinned = mask_staging.at(
          static_cast<std::size_t>(group[index]->id - 1)).ensure(mono.bytes);
      check_cuda(cudaMemcpyAsync(
          pinned,
          mono.data,
          mono.bytes,
          cudaMemcpyDeviceToHost,
          execution_stream));
      result.push_back(std::move(mask));
    }
    check_cuda(cudaStreamSynchronize(execution_stream));
    const auto copy_start = Clock::now();
    for (std::size_t index = 0; index < group.size(); ++index) {
      const auto* pinned = mask_staging.at(
          static_cast<std::size_t>(group[index]->id - 1)).ensure(
          result[index].mono8.size());
      std::memcpy(
          result[index].mono8.data(), pinned, result[index].mono8.size());
    }
    {
      std::lock_guard lock(timing_mutex);
      timings.host_mask_copy_ms += std::chrono::duration<double, std::milli>(
          Clock::now() - copy_start).count();
    }
    return result;
  }

  std::vector<ObjectMask> run_prompt_group(
      const std::map<std::string, DeviceTensor>& encoded, std::vector<ObjectState*> group,
      int prompt_count, int width, int height) {
    const int batch = padded_object_batch(static_cast<int>(group.size()));
    auto inputs = common_features(encoded, batch, stream);
    auto& prompt_engine = prompt_count == 1 ? point_prompt : box_prompt;
    const auto dtype = prompt_engine.tensor_dtype("point_coords");
    std::vector<float> host_coords(batch * prompt_count * 2);
    std::vector<int32_t> host_labels(batch * prompt_count, -1);
    for (int row = 0; row < batch; ++row) {
      const auto& p = group[std::min<int>(row, group.size() - 1)]->prompt;
      if (prompt_count == 1) {
        host_coords[row * 2] = p.x0 * 1024.0f / width;
        host_coords[row * 2 + 1] = p.y0 * 1024.0f / height;
        host_labels[row] = 1;
      } else {
        const int base = row * 4;
        host_coords[base] = p.x0 * 1024.0f / width;
        host_coords[base + 1] = p.y0 * 1024.0f / height;
        host_coords[base + 2] = p.x1 * 1024.0f / width;
        host_coords[base + 3] = p.y1 * 1024.0f / height;
        host_labels[row * 2] = 2; host_labels[row * 2 + 1] = 3;
      }
    }
    auto float_staging = allocate_tensor(
        {static_cast<int64_t>(host_coords.size())},
        nvinfer1::DataType::kFLOAT,
        stream);
    check_cuda(cudaMemcpyAsync(float_staging.data, host_coords.data(), float_staging.bytes,
                               cudaMemcpyHostToDevice, stream));
    auto coords = allocate_tensor({batch, prompt_count, 2}, dtype, stream);
    launch_float_conversion(static_cast<const float*>(float_staging.data), coords.data, dtype,
                            host_coords.size(), stream);
    auto labels = allocate_tensor(
        {batch, prompt_count}, nvinfer1::DataType::kINT32, stream);
    check_cuda(cudaMemcpyAsync(labels.data, host_labels.data(), labels.bytes, cudaMemcpyHostToDevice, stream));
    inputs["point_coords"] = std::move(coords);
    inputs["point_labels"] = std::move(labels);
    auto output = prompt_engine.run(inputs, profile_for_batch(batch), stream);
    for (std::size_t index = 0; index < group.size(); ++index) {
      save_outputs(*group[index], output, index, true);
      group[index]->pending = false;
    }
    return masks_from_output(group, output, width, height, stream);
  }

  std::vector<ObjectMask> run_track_group(
      const std::map<std::string, DeviceTensor>& encoded, std::vector<ObjectState*> group,
      const std::vector<SelectedState<std::shared_ptr<FrameState>>>& selections,
      int width, int height) {
    const auto slot = static_cast<std::size_t>(group.front()->id - 1);
    auto execution_stream = track_streams.at(slot);
    auto& track = *track_engines.at(slot);
    const int batch = padded_object_batch(static_cast<int>(group.size()));
    const int memories = static_cast<int>(selections.front().memories.size());
    const int pointers = static_cast<int>(selections.front().pointers.size());
    auto inputs = common_features(encoded, batch, execution_stream);
    inputs["image_position"] = repeat_batch(
        encoded.at("image_position"), batch, execution_stream);
    const auto dtype = track.tensor_dtype("mask_memory");
    auto memory = allocate_tensor(
        {memories, 4096, batch, 64}, dtype, execution_stream);
    auto memory_position = allocate_tensor(
        {memories, 4096, batch, 64}, dtype, execution_stream);
    auto object_pointers = allocate_tensor(
        {pointers, batch, 256}, dtype, execution_stream);
    std::vector<int64_t> temporal(memories * batch);
    std::vector<int64_t> distance(pointers * batch);
    for (int row = 0; row < batch; ++row) {
      const int source_row = std::min<int>(row, group.size() - 1);
      const auto& selected = selections[source_row];
      for (int index = 0; index < memories; ++index) {
        const auto& item = selected.memories[index];
        if (item.value->memory.dtype != dtype || item.value->memory_position.dtype != dtype)
          throw std::invalid_argument("memory dtype does not match track engine");
        temporal[index * batch + row] = item.position;
        launch_nchw_to_memory_bank(item.value->memory.data, memory.data, dtype, index, row,
                                   memories, batch, 64, 64, 64, execution_stream);
        launch_nchw_to_memory_bank(item.value->memory_position.data, memory_position.data,
                                   dtype, index, row, memories, batch, 64, 64, 64,
                                   execution_stream);
      }
      for (int index = 0; index < pointers; ++index) {
        const auto& item = selected.pointers[index];
        distance[index * batch + row] = item.position;
        const std::size_t offset = (static_cast<std::size_t>(index) * batch + row) * 256 * element_size(dtype);
        auto* destination = static_cast<std::byte*>(object_pointers.data) + offset;
        if (item.value->pointer.dtype == dtype) {
          check_cuda(cudaMemcpyAsync(destination, item.value->pointer.data,
                                     256 * element_size(dtype), cudaMemcpyDeviceToDevice,
                                     execution_stream));
        } else if (item.value->pointer.dtype == nvinfer1::DataType::kFLOAT) {
          launch_float_conversion(static_cast<const float*>(item.value->pointer.data),
                                  destination, dtype, 256, execution_stream);
        } else {
          throw std::invalid_argument("unsupported object pointer dtype conversion");
        }
      }
    }
    inputs["mask_memory"] = std::move(memory);
    inputs["mask_memory_position"] = std::move(memory_position);
    inputs["mask_temporal_position"] = device_from_i64(
        temporal, {memories, batch}, execution_stream);
    inputs["object_pointers"] = std::move(object_pointers);
    inputs["pointer_frame_distance"] = device_from_i64(
        distance, {pointers, batch}, execution_stream);
    check_cuda(cudaStreamWaitEvent(execution_stream, encoded_ready, 0));
    auto output = track.run(inputs, 0, execution_stream);
    for (std::size_t index = 0; index < group.size(); ++index)
      save_outputs(*group[index], output, index, false);
    return masks_from_output(
        group, output, width, height, execution_stream);
  }

  std::vector<ObjectMask> process(const std::uint8_t* image, int width, int height, std::size_t stride) {
    std::lock_guard lock(mutex);
    timings = {};
    const auto total_start = Clock::now();
    check_cuda(cudaEventRecord(gpu_start, stream));
    auto encoded = encode(image, width, height, stride);
    check_cuda(cudaEventRecord(encoder_end, stream));
    check_cuda(cudaEventRecord(encoded_ready, stream));
    std::vector<ObjectMask> masks;
    for (int prompt_count : {1, 2}) {
      std::vector<ObjectState*> group;
      for (auto& object : objects)
        if (object.pending && (object.prompt.kind == PromptKind::Point ? 1 : 2) == prompt_count)
          group.push_back(&object);
      if (!group.empty()) {
        auto output = run_prompt_group(encoded, group, prompt_count, width, height);
        masks.insert(masks.end(), std::make_move_iterator(output.begin()), std::make_move_iterator(output.end()));
      }
    }

    std::map<std::pair<int, int>, std::vector<std::pair<ObjectState*, SelectedState<std::shared_ptr<FrameState>>>>> buckets;
    for (auto& object : objects) {
      if (object.conditioning.contains(frame_index)) continue;
      auto selected = select_state(frame_index, frame_index + 1, object.conditioning,
                                   object.non_conditioning);
      if (selected.memories.empty() || selected.pointers.empty())
        throw std::runtime_error("tracking object has no conditioning state");
      buckets[{selected.memories.size(), selected.pointers.size()}].push_back({&object, std::move(selected)});
    }
    std::vector<std::future<std::vector<ObjectMask>>> track_futures;
    const auto collect_futures = [&]() {
      for (auto& future : track_futures) {
        auto output = future.get();
        masks.insert(
            masks.end(),
            std::make_move_iterator(output.begin()),
            std::make_move_iterator(output.end()));
      }
      track_futures.clear();
    };
    for (auto& [key, entries] : buckets) {
      for (auto& entry : entries) {
        auto* object = entry.first;
        auto selection = std::move(entry.second);
        track_futures.push_back(std::async(
            std::launch::async,
            [this, &encoded, object, selection = std::move(selection),
             width, height]() mutable {
              std::vector<ObjectState*> group{object};
              std::vector<SelectedState<std::shared_ptr<FrameState>>> selections;
              selections.push_back(std::move(selection));
              return run_track_group(
                  encoded, std::move(group), selections, width, height);
            }));
        if (static_cast<int>(track_futures.size()) == track_concurrency)
          collect_futures();
      }
    }
    collect_futures();
    check_cuda(cudaEventRecord(gpu_end, stream));
    check_cuda(cudaEventSynchronize(gpu_end));
    float encoder_ms = 0.0f;
    float tail_ms = 0.0f;
    float total_gpu_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&encoder_ms, gpu_start, encoder_end));
    check_cuda(cudaEventElapsedTime(&tail_ms, encoder_end, gpu_end));
    check_cuda(cudaEventElapsedTime(&total_gpu_ms, gpu_start, gpu_end));
    timings.encoder_gpu_ms = encoder_ms;
    timings.tail_gpu_ms = tail_ms;
    timings.gpu_total_ms = total_gpu_ms;
    timings.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    ++frame_index;
    return masks;
  }
};

Tracker::Tracker(
    const std::string& bundle, const std::string& precision, int maximum,
    int track_concurrency)
    : impl_(std::make_unique<Impl>(
          bundle, precision, maximum, track_concurrency)) {}
Tracker::~Tracker() = default;

int Tracker::add_object(const Prompt& prompt) {
  std::lock_guard lock(impl_->mutex);
  if (static_cast<int>(impl_->objects.size()) >= impl_->maximum_objects)
    throw std::runtime_error("maximum object count reached");
  const int id = impl_->next_id++;
  impl_->objects.push_back({id, prompt});
  return id;
}

void Tracker::reset() {
  std::lock_guard lock(impl_->mutex);
  impl_->objects.clear();
  impl_->frame_index = 0;
  impl_->next_id = 1;
}

std::vector<ObjectMask> Tracker::process_rgb8(
    const std::uint8_t* image, int width, int height, std::size_t stride) {
  if (!image || width < 1 || height < 1 || stride < static_cast<std::size_t>(width * 3))
    throw std::invalid_argument("invalid RGB8 frame");
  return impl_->process(image, width, height, stride);
}

TrackerTimings Tracker::last_timings() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->timings;
}

}  // namespace sam2_trt
