#include "sam2_trt_ros/sam2_trt_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(make_sam2_trt_node());
  rclcpp::shutdown();
  return 0;
}
