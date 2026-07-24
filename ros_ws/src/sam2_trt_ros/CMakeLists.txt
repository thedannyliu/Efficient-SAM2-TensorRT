cmake_minimum_required(VERSION 3.16)
project(sam2_trt_ros)

set(CMAKE_CXX_STANDARD 20)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sam2_trt_msgs REQUIRED)
find_package(sam2_trt_core REQUIRED CONFIG)

add_library(sam2_trt_component SHARED src/sam2_trt_node.cpp)
target_include_directories(sam2_trt_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sam2_trt_component sam2_trt::sam2_trt_core)
ament_target_dependencies(sam2_trt_component
  rclcpp rclcpp_components sensor_msgs std_msgs std_srvs sam2_trt_msgs)
rclcpp_components_register_nodes(sam2_trt_component "Sam2TrtNode")

add_executable(sam2_trt_node src/sam2_trt_main.cpp)
target_link_libraries(sam2_trt_node sam2_trt_component)
ament_target_dependencies(sam2_trt_node rclcpp)

install(TARGETS sam2_trt_component sam2_trt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS scripts/sam2_trt_interactive_viewer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sam2_trt_interactive_viewer)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
