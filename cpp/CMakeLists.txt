cmake_minimum_required(VERSION 3.24)
project(sam2_trt_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/local/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/library were not found")
endif()

add_library(sam2_trt_core SHARED
  src/engine.cpp
  src/tracker.cpp
  src/kernels.cu
)
target_include_directories(sam2_trt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sam2_trt_core PUBLIC CUDA::cudart ${TENSORRT_LIBRARY})
target_compile_options(sam2_trt_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_executable(state_selection_test tests/state_selection_test.cpp)
target_include_directories(state_selection_test PRIVATE include)
add_executable(fused_gather_test tests/fused_gather_test.cu)
target_link_libraries(fused_gather_test PRIVATE sam2_trt_core)
enable_testing()
add_test(NAME state_selection_test COMMAND state_selection_test)
add_test(NAME fused_gather_test COMMAND fused_gather_test)

include(CMakePackageConfigHelpers)
install(TARGETS sam2_trt_core EXPORT sam2_trt_coreTargets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT sam2_trt_coreTargets
  FILE sam2_trt_coreTargets.cmake NAMESPACE sam2_trt::
  DESTINATION lib/cmake/sam2_trt_core)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sam2_trt_coreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sam2_trt_coreConfig.cmake
  INSTALL_DESTINATION lib/cmake/sam2_trt_core)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sam2_trt_coreConfig.cmake
  DESTINATION lib/cmake/sam2_trt_core)
