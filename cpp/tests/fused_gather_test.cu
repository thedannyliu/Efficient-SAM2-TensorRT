#include "sam2_trt/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

}  // namespace

int main() {
  constexpr int memories = 2;
  constexpr int batch = 2;
  constexpr int tokens = 3;
  constexpr int channels = 2;
  constexpr int source_count = memories * batch;
  constexpr int values_per_source = tokens * channels;

  std::vector<std::vector<__half>> host_sources(source_count);
  std::vector<void*> device_sources(source_count, nullptr);
  std::vector<std::uint64_t> pointers(source_count);
  for (int source = 0; source < source_count; ++source) {
    host_sources[source].resize(values_per_source);
    for (int index = 0; index < values_per_source; ++index)
      host_sources[source][index] =
          __float2half(static_cast<float>(source * 100 + index));
    if (!check(
            cudaMalloc(
                &device_sources[source],
                values_per_source * sizeof(__half)),
            "cudaMalloc source"))
      return 1;
    if (!check(
            cudaMemcpy(
                device_sources[source], host_sources[source].data(),
                values_per_source * sizeof(__half),
                cudaMemcpyHostToDevice),
            "cudaMemcpy source"))
      return 1;
    pointers[source] =
        reinterpret_cast<std::uintptr_t>(device_sources[source]);
  }

  std::uint64_t* device_pointers = nullptr;
  __half* device_output = nullptr;
  const int output_count = memories * tokens * batch * channels;
  if (!check(
          cudaMalloc(&device_pointers, pointers.size() * sizeof(std::uint64_t)),
          "cudaMalloc pointers") ||
      !check(
          cudaMalloc(&device_output, output_count * sizeof(__half)),
          "cudaMalloc output") ||
      !check(
          cudaMemcpy(
              device_pointers, pointers.data(),
              pointers.size() * sizeof(std::uint64_t),
              cudaMemcpyHostToDevice),
          "cudaMemcpy pointers"))
    return 1;

  sam2_trt::launch_gather_memory_bank(
      device_pointers, device_output, nvinfer1::DataType::kHALF, memories,
      batch, tokens, channels, nullptr);
  std::vector<__half> output(output_count);
  if (!check(
          cudaMemcpy(
              output.data(), device_output, output.size() * sizeof(__half),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy output"))
    return 1;

  for (int memory = 0; memory < memories; ++memory)
    for (int token = 0; token < tokens; ++token)
      for (int row = 0; row < batch; ++row)
        for (int channel = 0; channel < channels; ++channel) {
          const int output_index =
              ((memory * tokens + token) * batch + row) * channels + channel;
          const int source_index = memory * batch + row;
          const float expected =
              static_cast<float>(source_index * 100 + token * channels + channel);
          if (__half2float(output[output_index]) != expected) {
            std::cerr << "mismatch at " << output_index << '\n';
            return 1;
          }
        }

  for (auto* pointer : device_sources) cudaFree(pointer);
  cudaFree(device_pointers);
  cudaFree(device_output);
  return 0;
}
