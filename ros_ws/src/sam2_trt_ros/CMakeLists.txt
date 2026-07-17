cmake_minimum_required(VERSION 3.16)
project(sam2_trt_ros)

set(CMAKE_CXX_STANDARD 20)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sam2_trt_msgs REQUIRED)
find_package(sam2_trt_core REQUIRED CONFIG)

add_executable(sam2_trt_node src/sam2_trt_node.cpp)
target_link_libraries(sam2_trt_node sam2_trt::sam2_trt_core)
ament_target_dependencies(sam2_trt_node
  rclcpp sensor_msgs std_msgs std_srvs sam2_trt_msgs)
install(TARGETS sam2_trt_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
