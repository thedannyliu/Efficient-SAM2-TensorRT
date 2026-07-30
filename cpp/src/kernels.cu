#include "sam2_trt/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace sam2_trt {
namespace {

template <class T> __device__ T convert(float value);
template <> __device__ float convert<float>(float value) { return value; }
template <> __device__ __half convert<__half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 convert<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <class T> __device__ float as_float(T value);
template <> __device__ float as_float<float>(float value) { return value; }
template <> __device__ float as_float<__half>(__half value) { return __half2float(value); }
template <> __device__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class T>
__global__ void preprocess_kernel(
    const std::uint8_t* input, int width, int height, std::size_t stride, T* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 1024 || y >= 1024) return;
  const float source_x = (x + 0.5f) * width / 1024.0f - 0.5f;
  const float source_y = (y + 0.5f) * height / 1024.0f - 0.5f;
  const int base_x = static_cast<int>(floorf(source_x));
  const int base_y = static_cast<int>(floorf(source_y));
  const int x0 = max(0, min(width - 1, base_x));
  const int y0 = max(0, min(height - 1, base_y));
  const int x1 = max(0, min(width - 1, base_x + 1));
  const int y1 = max(0, min(height - 1, base_y + 1));
  const float dx = source_x - floorf(source_x);
  const float dy = source_y - floorf(source_y);
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float inverse_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
  for (int channel = 0; channel < 3; ++channel) {
    const float a = input[y0 * stride + 3 * x0 + channel];
    const float b = input[y0 * stride + 3 * x1 + channel];
    const float c = input[y1 * stride + 3 * x0 + channel];
    const float d = input[y1 * stride + 3 * x1 + channel];
    const float value = ((a + dx * (b - a)) + dy * ((c + dx * (d - c)) - (a + dx * (b - a)))) / 255.0f;
    output[(channel * 1024 + y) * 1024 + x] = convert<T>((value - mean[channel]) * inverse_std[channel]);
  }
}

template <class T>
__global__ void mask_kernel(const T* input, int sw, int sh, std::uint8_t* output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float source_x = (x + 0.5f) * sw / width - 0.5f;
  const float source_y = (y + 0.5f) * sh / height - 0.5f;
  const int base_x = static_cast<int>(floorf(source_x));
  const int base_y = static_cast<int>(floorf(source_y));
  const int x0 = max(0, min(sw - 1, base_x));
  const int y0 = max(0, min(sh - 1, base_y));
  const int x1 = max(0, min(sw - 1, base_x + 1));
  const int y1 = max(0, min(sh - 1, base_y + 1));
  const float dx = source_x - floorf(source_x);
  const float dy = source_y - floorf(source_y);
  const float a = as_float(input[y0 * sw + x0]);
  const float b = as_float(input[y0 * sw + x1]);
  const float c = as_float(input[y1 * sw + x0]);
  const float d = as_float(input[y1 * sw + x1]);
  const float top = a + dx * (b - a);
  const float value = top + dy * ((c + dx * (d - c)) - top);
  output[y * width + x] = value > 0.0f ? 255 : 0;
}

template <class T>
__global__ void preprocess_mask_kernel(
    const std::uint8_t* input, int width, int height, std::size_t stride,
    T* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 1024 || y >= 1024) return;
  const float source_x = (x + 0.5f) * width / 1024.0f - 0.5f;
  const float source_y = (y + 0.5f) * height / 1024.0f - 0.5f;
  const int base_x = static_cast<int>(floorf(source_x));
  const int base_y = static_cast<int>(floorf(source_y));
  const int x0 = max(0, min(width - 1, base_x));
  const int y0 = max(0, min(height - 1, base_y));
  const int x1 = max(0, min(width - 1, base_x + 1));
  const int y1 = max(0, min(height - 1, base_y + 1));
  const float dx = source_x - floorf(source_x);
  const float dy = source_y - floorf(source_y);
  const float a = input[y0 * stride + x0];
  const float b = input[y0 * stride + x1];
  const float c = input[y1 * stride + x0];
  const float d = input[y1 * stride + x1];
  const float top = a + dx * (b - a);
  const float value = top + dy * ((c + dx * (d - c)) - top);
  output[y * 1024 + x] = convert<T>(value >= 127.5f ? 1.0f : 0.0f);
}

template <class T>
__global__ void memory_kernel(
    const T* source, T* destination, int memory_index, int batch_index,
    int memory_count, int batch, int channels, int height, int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int spatial = height * width;
  if (index >= channels * spatial) return;
  const int channel = index / spatial;
  const int token = index % spatial;
  const std::size_t target = (((static_cast<std::size_t>(memory_index) * spatial + token) * batch + batch_index) * channels + channel);
  destination[target] = source[channel * spatial + token];
}

template <class T>
__global__ void pack_memory_bank_kernel(
    const T* source, T* destination, int memory_index, int batch_index,
    int batch, int tokens, int channels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * channels) return;
  const int token = index / channels;
  const int channel = index % channels;
  const std::size_t target =
      (((static_cast<std::size_t>(memory_index) * tokens + token) * batch +
        batch_index) *
       channels +
       channel);
  destination[target] = source[index];
}

template <class T>
__global__ void conversion_kernel(const float* input, T* output, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = convert<T>(input[index]);
}

template <class T>
void preprocess(const std::uint8_t* input, int width, int height, std::size_t stride, void* output, cudaStream_t stream) {
  preprocess_kernel<<<dim3(32, 32), dim3(32, 32), 0, stream>>>(input, width, height, stride, static_cast<T*>(output));
}

template <class T>
void preprocess_mask(
    const std::uint8_t* input, int width, int height, std::size_t stride,
    void* output, cudaStream_t stream) {
  preprocess_mask_kernel<<<dim3(32, 32), dim3(32, 32), 0, stream>>>(
      input, width, height, stride, static_cast<T*>(output));
}

}  // namespace

void launch_preprocess_rgb8(const std::uint8_t* input, int width, int height, std::size_t stride, void* output, nvinfer1::DataType dtype, cudaStream_t stream) {
  if (dtype == nvinfer1::DataType::kFLOAT) preprocess<float>(input, width, height, stride, output, stream);
  else if (dtype == nvinfer1::DataType::kHALF) preprocess<__half>(input, width, height, stride, output, stream);
  else if (dtype == nvinfer1::DataType::kBF16) preprocess<__nv_bfloat16>(input, width, height, stride, output, stream);
  else throw std::invalid_argument("unsupported preprocess dtype");
}

void launch_preprocess_mono8_mask(
    const std::uint8_t* input, int width, int height, std::size_t stride,
    void* output, nvinfer1::DataType dtype, cudaStream_t stream) {
  if (dtype == nvinfer1::DataType::kFLOAT)
    preprocess_mask<float>(input, width, height, stride, output, stream);
  else if (dtype == nvinfer1::DataType::kHALF)
    preprocess_mask<__half>(input, width, height, stride, output, stream);
  else if (dtype == nvinfer1::DataType::kBF16)
    preprocess_mask<__nv_bfloat16>(
        input, width, height, stride, output, stream);
  else
    throw std::invalid_argument("unsupported mask preprocess dtype");
}

void launch_mask_to_mono8(const void* logits, nvinfer1::DataType dtype, int sw, int sh, std::uint8_t* output, int width, int height, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  if (dtype == nvinfer1::DataType::kFLOAT) mask_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(logits), sw, sh, output, width, height);
  else if (dtype == nvinfer1::DataType::kHALF) mask_kernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(logits), sw, sh, output, width, height);
  else if (dtype == nvinfer1::DataType::kBF16) mask_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits), sw, sh, output, width, height);
  else throw std::invalid_argument("unsupported mask dtype");
}

void launch_nchw_to_memory_bank(const void* source, void* destination, nvinfer1::DataType dtype, int memory_index, int batch_index, int memory_count, int batch, int channels, int height, int width, cudaStream_t stream) {
  const int count = channels * height * width;
  const int blocks = (count + 255) / 256;
  if (dtype == nvinfer1::DataType::kFLOAT) memory_kernel<<<blocks, 256, 0, stream>>>(static_cast<const float*>(source), static_cast<float*>(destination), memory_index, batch_index, memory_count, batch, channels, height, width);
  else if (dtype == nvinfer1::DataType::kHALF) memory_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __half*>(source), static_cast<__half*>(destination), memory_index, batch_index, memory_count, batch, channels, height, width);
  else if (dtype == nvinfer1::DataType::kBF16) memory_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(source), static_cast<__nv_bfloat16*>(destination), memory_index, batch_index, memory_count, batch, channels, height, width);
  else throw std::invalid_argument("unsupported memory dtype");
}

void launch_pack_memory_bank(
    const void* source, void* destination, nvinfer1::DataType dtype,
    int memory_index, int batch_index, int batch, int tokens, int channels,
    cudaStream_t stream) {
  const int count = tokens * channels;
  const int blocks = (count + 255) / 256;
  if (dtype == nvinfer1::DataType::kFLOAT)
    pack_memory_bank_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(source), static_cast<float*>(destination),
        memory_index, batch_index, batch, tokens, channels);
  else if (dtype == nvinfer1::DataType::kHALF)
    pack_memory_bank_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __half*>(source), static_cast<__half*>(destination),
        memory_index, batch_index, batch, tokens, channels);
  else if (dtype == nvinfer1::DataType::kBF16)
    pack_memory_bank_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source),
        static_cast<__nv_bfloat16*>(destination), memory_index, batch_index,
        batch, tokens, channels);
  else
    throw std::invalid_argument("unsupported memory dtype");
}

void launch_float_conversion(const float* input, void* output, nvinfer1::DataType dtype, std::size_t count, cudaStream_t stream) {
  const int blocks = static_cast<int>((count + 255) / 256);
  if (dtype == nvinfer1::DataType::kFLOAT) cudaMemcpyAsync(output, input, count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  else if (dtype == nvinfer1::DataType::kHALF) conversion_kernel<<<blocks, 256, 0, stream>>>(input, static_cast<__half*>(output), count);
  else if (dtype == nvinfer1::DataType::kBF16) conversion_kernel<<<blocks, 256, 0, stream>>>(input, static_cast<__nv_bfloat16*>(output), count);
  else throw std::invalid_argument("unsupported conversion dtype");
}

}  // namespace sam2_trt
