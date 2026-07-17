#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace sam2_trt {

void launch_preprocess_rgb8(
    const std::uint8_t* input, int width, int height, std::size_t row_stride,
    void* output, nvinfer1::DataType dtype, cudaStream_t stream);
void launch_mask_to_mono8(
    const void* logits, nvinfer1::DataType dtype, int source_width, int source_height,
    std::uint8_t* output, int width, int height, cudaStream_t stream);
void launch_nchw_to_memory_bank(
    const void* source, void* destination, nvinfer1::DataType dtype,
    int memory_index, int batch_index, int memory_count, int batch, int channels,
    int height, int width, cudaStream_t stream);
void launch_float_conversion(
    const float* input, void* output, nvinfer1::DataType dtype,
    std::size_t count, cudaStream_t stream);

}  // namespace sam2_trt
