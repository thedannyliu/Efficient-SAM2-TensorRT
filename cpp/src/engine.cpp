#include "sam2_trt/engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace sam2_trt {

class Engine::Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

void CudaDeleter::operator()(void* pointer) const noexcept {
  if (stream) {
    cudaFreeAsync(pointer, stream);
  } else {
    cudaFree(pointer);
  }
}

std::size_t element_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kFP8: return 1;
    default: throw std::invalid_argument("unsupported TensorRT data type");
  }
}

DeviceTensor allocate_tensor(
    std::vector<int64_t> shape, nvinfer1::DataType dtype, cudaStream_t stream) {
  std::size_t count = 1;
  for (auto dimension : shape) {
    if (dimension < 0) throw std::invalid_argument("cannot allocate a dynamic tensor dimension");
    count *= static_cast<std::size_t>(dimension);
  }
  const std::size_t bytes = count * element_size(dtype);
  void* pointer = nullptr;
  const auto status = stream ? cudaMallocAsync(&pointer, bytes, stream) : cudaMalloc(&pointer, bytes);
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
  return {
      std::shared_ptr<void>(pointer, CudaDeleter{stream}),
      pointer,
      std::move(shape),
      dtype,
      bytes,
  };
}

static std::vector<char> read_plan(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open TensorRT plan: " + path);
  const auto size = stream.tellg();
  std::vector<char> payload(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(payload.data(), size);
  if (!stream) throw std::runtime_error("cannot read TensorRT plan: " + path);
  return payload;
}

Engine::Engine(
    const std::string& plan_path, bool profile_zero_only, int context_copies)
    : logger_(std::make_unique<Logger>()),
      context_copies_(context_copies) {
  if (context_copies_ < 1)
    throw std::invalid_argument("context_copies must be positive");
  const auto payload = read_plan(plan_path);
  runtime_ = nvinfer1::createInferRuntime(*logger_);
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_ = runtime_->deserializeCudaEngine(payload.data(), payload.size());
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed for " + plan_path);
  profile_count_ = profile_zero_only
      ? 1
      : std::max(1, engine_->getNbOptimizationProfiles());
  contexts_.reserve(profile_count_ * context_copies_);
  for (int profile = 0; profile < profile_count_; ++profile) {
    for (int copy = 0; copy < context_copies_; ++copy) {
      auto* context = engine_->createExecutionContext();
      if (!context) throw std::runtime_error("createExecutionContext failed");
      contexts_.push_back(context);
    }
  }
}

Engine::~Engine() {
  for (auto* context : contexts_) delete context;
  delete engine_;
  delete runtime_;
}

nvinfer1::DataType Engine::tensor_dtype(const std::string& name) const {
  return engine_->getTensorDataType(name.c_str());
}

static nvinfer1::Dims to_dims(const std::vector<int64_t>& shape) {
  if (shape.size() > nvinfer1::Dims::MAX_DIMS) throw std::invalid_argument("too many tensor dimensions");
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(shape.size());
  for (int index = 0; index < dims.nbDims; ++index) dims.d[index] = shape[index];
  return dims;
}

static std::vector<int64_t> from_dims(const nvinfer1::Dims& dims) {
  std::vector<int64_t> shape(dims.nbDims);
  for (int index = 0; index < dims.nbDims; ++index) shape[index] = dims.d[index];
  return shape;
}

std::map<std::string, DeviceTensor> Engine::run(
    const std::map<std::string, DeviceTensor>& inputs, int profile, cudaStream_t stream) {
  std::map<std::string, DeviceTensor> outputs;
  run_into(inputs, profile, stream, outputs);
  return outputs;
}

void Engine::run_into(
    const std::map<std::string, DeviceTensor>& inputs, int profile,
    cudaStream_t stream, std::map<std::string, DeviceTensor>& outputs,
    int context_copy) {
  if (profile < 0 || profile >= profile_count_)
    throw std::out_of_range("invalid TensorRT optimization profile");
  if (context_copy < 0 || context_copy >= context_copies_)
    throw std::out_of_range("invalid TensorRT execution context copy");
  auto* context = contexts_[
      static_cast<std::size_t>(profile * context_copies_ + context_copy)];
  if (profile_count_ > 1 &&
      !context->setOptimizationProfileAsync(profile, stream))
    throw std::runtime_error("setOptimizationProfileAsync failed");

  for (const auto& [name, tensor] : inputs) {
    if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
      throw std::invalid_argument("unknown engine input: " + name);
    if (engine_->getTensorDataType(name.c_str()) != tensor.dtype)
      throw std::invalid_argument("input dtype mismatch: " + name);
    if (!context->setInputShape(name.c_str(), to_dims(tensor.shape)))
      throw std::invalid_argument("input shape rejected: " + name);
    if (!context->setTensorAddress(name.c_str(), tensor.data))
      throw std::runtime_error("setTensorAddress failed: " + name);
  }
  if (!context->allInputDimensionsSpecified()) throw std::invalid_argument("not all input dimensions are specified");

  for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
    const char* raw_name = engine_->getIOTensorName(index);
    if (engine_->getTensorIOMode(raw_name) != nvinfer1::TensorIOMode::kOUTPUT) continue;
    std::string name(raw_name);
    const auto shape = from_dims(context->getTensorShape(raw_name));
    const auto dtype = engine_->getTensorDataType(raw_name);
    auto existing = outputs.find(name);
    if (existing == outputs.end() || existing->second.shape != shape ||
        existing->second.dtype != dtype) {
      outputs[name] = allocate_tensor(shape, dtype, stream);
      existing = outputs.find(name);
    }
    if (!context->setTensorAddress(raw_name, existing->second.data))
      throw std::runtime_error("setTensorAddress failed: " + name);
  }
  if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
}

}  // namespace sam2_trt
