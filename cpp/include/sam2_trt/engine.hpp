#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sam2_trt {

struct CudaDeleter {
  cudaStream_t stream{};
  void operator()(void* pointer) const noexcept;
};

struct DeviceTensor {
  std::shared_ptr<void> storage;
  void* data{};
  std::vector<int64_t> shape;
  nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
  std::size_t bytes{};
};

DeviceTensor allocate_tensor(
    std::vector<int64_t> shape, nvinfer1::DataType dtype, cudaStream_t stream);
std::size_t element_size(nvinfer1::DataType dtype);

class Engine {
 public:
  explicit Engine(
      const std::string& plan_path, bool profile_zero_only = false,
      int context_copies = 1, bool cuda_graph = false);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  std::map<std::string, DeviceTensor> run(
      const std::map<std::string, DeviceTensor>& inputs, int profile, cudaStream_t stream);
  void run_into(
      const std::map<std::string, DeviceTensor>& inputs, int profile,
      cudaStream_t stream, std::map<std::string, DeviceTensor>& outputs,
      int context_copy = 0);
  nvinfer1::DataType tensor_dtype(const std::string& name) const;

 private:
  class Logger;
  struct GraphCache {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<std::uintptr_t> signature;
    bool disabled{};
  };
  static void clear_graph(GraphCache& cache);
  std::unique_ptr<Logger> logger_;
  nvinfer1::IRuntime* runtime_{};
  nvinfer1::ICudaEngine* engine_{};
  std::vector<nvinfer1::IExecutionContext*> contexts_;
  int profile_count_{};
  int context_copies_{};
  bool cuda_graph_{};
  std::vector<GraphCache> graph_caches_;
};

}  // namespace sam2_trt
